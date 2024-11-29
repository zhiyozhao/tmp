import os.path as osp
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, AutoTokenizer, AutoFeatureExtractor


class VQADataset(Dataset):
    def __init__(
        self,
        data_root,
        csv_file,
        answer_space_file,
    ):
        """
        Args:
            data_root (str): Directory containing the 'images' folder.
            csv_file (str): Path to the CSV file (train/eval).
            answer_space_file (str): Path to the answer space file (answer_space.txt).
        """
        self.data_root = data_root
        self.img_dir = osp.join(data_root, "images")
        self.csv_file = osp.join(data_root, csv_file)
        self.answer_space_file = osp.join(data_root, answer_space_file)

        self.load_data()

    def load_data(self):
        # Load answer space (mapping answers to indices)

        # sorted to gurantee order between train and test
        with open(self.answer_space_file, "r") as f:
            self.answer_space = sorted([line.strip() for line in f.readlines()])
        self.answer_to_idx = {
            answer: idx for idx, answer in enumerate(self.answer_space)
        }

        # Load dataset
        self.df = pd.read_csv(self.csv_file)
        self.df["answer_idx"] = self.df["answer"].apply(self.get_first_answer_index)

    def get_first_answer_index(self, answer):
        """Use only the first answer when there are multiple answers."""
        first_answer = answer.split(",")[0].strip()  # Get the first answer
        assert (
            first_answer in self.answer_to_idx
        ), f"Answer {first_answer} not in answer space"

        return self.answer_to_idx[first_answer]  # Map to index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image path and question-answer pair
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        img_path = osp.join(self.img_dir, f"{img_id}.png")

        # Open image and question text
        image = Image.open(img_path).convert("RGB")
        question = row["question"]
        answer_idx = row["answer_idx"]

        return image, question, answer_idx


def build_collate_fn(
    combined_processor=None, image_processor=None, text_processor=None
):
    def collate_fn(batch):
        # Prepare lists for images, questions, and their corresponding answers
        images = []
        questions = []
        answer_indices = []
        for image, question, answer_idx in batch:
            images.append(image)
            questions.append(question)
            answer_indices.append(answer_idx)

        if combined_processor is not None:
            inputs = combined_processor(
                images=images,
                text=questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            image_inputs = image_processor(images, return_tensors="pt", padding=True)
            text_inputs = text_processor(
                questions, return_tensors="pt", padding=True, truncation=True
            )

            # Combine the processed outputs
            inputs = {
                "pixel_values": image_inputs["pixel_values"],
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
            }

        return (
            inputs["pixel_values"],
            inputs["input_ids"],
            inputs["attention_mask"],
            torch.tensor(answer_indices),
        )

    return collate_fn


# Example: Usage for DataLoader
if __name__ == "__main__":
    data_root = "../input/visual-question-answering-computer-vision-nlp/dataset"
    train_csv = "data_train.csv"
    val_csv = "data_eval.csv"
    answer_space_file = "answer_space.txt"

    # Instantiate dataset
    train_dataset = VQADataset(data_root, train_csv, answer_space_file)
    val_dataset = VQADataset(data_root, val_csv, answer_space_file)
    assert train_dataset.answer_to_idx == val_dataset.answer_to_idx

    # === TEST CASE 1: Combined Processor (CLIPProcessor) ===
    clip_name = "openai/clip-vit-base-patch32"
    combined_processor = CLIPProcessor.from_pretrained(clip_name)

    collate_fn_clip = build_collate_fn(combined_processor=combined_processor)

    train_dataloader_clip = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_clip
    )
    val_dataloader_clip = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_clip
    )

    # Inspect a batch with CLIPProcessor
    print("=== Testing CLIPProcessor ===")
    clip_sample = next(iter(train_dataloader_clip))
    print(f"{clip_sample[0].shape}, {clip_sample[0].dtype}")
    print(f"{clip_sample[1].shape}, {clip_sample[1].dtype}")
    print(f"{clip_sample[2].shape}, {clip_sample[2].dtype}")
    print(f"{clip_sample[3].shape}, {clip_sample[3].dtype}")
    print(clip_sample[2][0])

    # === TEST CASE 2: Separate Image and Text Processors ===
    vit_name = "google/vit-base-patch16-224-in21k"  # Vision Transformer
    bert_name = "bert-base-uncased"  # BERT for text

    image_processor = AutoFeatureExtractor.from_pretrained(vit_name)
    text_processor = AutoTokenizer.from_pretrained(bert_name)

    collate_fn_separate = build_collate_fn(
        image_processor=image_processor, text_processor=text_processor
    )

    train_dataloader_separate = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_separate
    )
    val_dataloader_separate = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_separate
    )

    # Inspect a batch with separate processors
    print("\n=== Testing Separate Image and Text Processors ===")
    separate_sample = next(iter(train_dataloader_separate))
    print(f"{separate_sample[0].shape}, {separate_sample[0].dtype}")
    print(f"{separate_sample[1].shape}, {separate_sample[1].dtype}")
    print(f"{separate_sample[2].shape}, {separate_sample[2].dtype}")
    print(f"{separate_sample[3].shape}, {separate_sample[3].dtype}")
    print(separate_sample[2][0])
