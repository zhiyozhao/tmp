import cv2


def image_t(im, scale=1.0, rot=45, trans=(50, -50)):
    h, w = im.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, rot, scale)
    rotation_matrix[0, 2] += trans[0]
    rotation_matrix[1, 2] += trans[1]

    result = cv2.warpAffine(im, rotation_matrix, (w, h))

    return result


if __name__ == "__main__":
    im = cv2.imread("./misc/pearl.jpeg")

    scale = 0.5
    rot = 45
    trans = (50, -50)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite("./results/affine_result.png", result)
