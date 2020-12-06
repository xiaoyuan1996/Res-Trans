import numpy as np
import os
import cv2

def Data_augmentation(image_root, output_root):
    """
    image_root: the source data root
    output_root: the output images file root
    """

    def add_noise(image):
        trigle = np.random.randint(1, 10)
        if trigle >= 6:
            row, col = image.shape
            for _ in range(5000):
                x = np.random.randint(0, row)
                y = np.random.randint(0, col)
                image[x, y] = 255

            return image
        else:
            return image

    for cls in ['1/', '2/', '3/', '4/']:
        image_files = os.listdir(image_root + cls)
        for file in image_files:
            path = image_root + cls + file
            image = cv2.imread(path)
            h, w, _ = image.shape

            image_crop1 = add_noise(cv2.resize(image[0:round(0.75 * h), 0:round(0.75 * w), :], 278))
            image_crop2 = add_noise(cv2.resize(image[0:round(0.75 * h), round(0.25 * w):w, :], 278))
            image_crop3 = add_noise(cv2.resize(image[round(0.25 * h):h, 0:round(0.75 * w), :], 278))
            image_crop4 = add_noise(cv2.resize(image[round(0.25 * h):h, round(0.25 * w):w, :], 278))
            image = add_noise(cv2.resize(image, 278))

            cv2.imwrite(output_root + file.strip('.jpg') + '_0.jpg', image)
            cv2.imwrite(output_root + file.strip('.jpg') + '_1.jpg', image_crop1)
            cv2.imwrite(output_root + file.strip('.jpg') + '_2.jpg', image_crop2)
            cv2.imwrite(output_root + file.strip('.jpg') + '_3.jpg', image_crop3)
            cv2.imwrite(output_root + file.strip('.jpg') + '_4.jpg', image_crop4)

