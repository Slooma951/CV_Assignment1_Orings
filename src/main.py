import cv2
import numpy as np
import os
import time


def load_images(folder):
    images = []
    filenames = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            images.append(img)
            filenames.append(filename)

    return images, filenames


def main():

    image_folder = "Orings"
    output_folder = "outputs"

    images, filenames = load_images(image_folder)

    for img, filename in zip(images, filenames):

        start_time = time.time()

        # PROCESSING PIPELINE WILL GO HERE

        result = "TEST"

        end_time = time.time()
        processing_time = end_time - start_time

        cv2.putText(img,
                    f"{result} | {processing_time:.4f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)

        print("Processed:", filename)


if __name__ == "__main__":
    main()