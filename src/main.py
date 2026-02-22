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

        #GRAYSCALE
        gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                b = img[i, j, 0]
                g = img[i, j, 1]
                r = img[i, j, 2]
                gray[i, j] = int((int(r) + int(g) + int(b)) / 3)

        # HISTOGRAM 
        hist = np.zeros(256)

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                intensity = gray[i, j]
                hist[intensity] += 1

        #AUTOMATIC THRESHOLD (Two-Peak Method)
        peak1 = np.argmax(hist[:128])
        peak2 = np.argmax(hist[128:]) + 128
        threshold = int((peak1 + peak2) / 2)

        #APPLY THRESHOLD
        binary = np.zeros_like(gray)

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i, j] > threshold:
                    binary[i, j] = 255
                else:
                    binary[i, j] = 0

        
        cv2.imwrite(os.path.join(output_folder, "binary_" + filename), binary)

        result = "THRESHOLD DONE"

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