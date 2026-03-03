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


# ---------------- DILATION ----------------
def dilate(binary):
    rows, cols = binary.shape
    output = np.zeros_like(binary)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.any(binary[i-1:i+2, j-1:j+2] == 255):
                output[i, j] = 255

    return output


# ---------------- EROSION ----------------
def erode(binary):
    rows, cols = binary.shape
    output = np.zeros_like(binary)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.all(binary[i-1:i+2, j-1:j+2] == 255):
                output[i, j] = 255

    return output


def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_folder = os.path.join(base_dir, "Orings")
    output_folder = os.path.join(base_dir, "outputs")

    images, filenames = load_images(image_folder)

    for img, filename in zip(images, filenames):

        start_time = time.time()

        # ---------- GRAYSCALE ----------
        gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                b = img[x, y, 0]
                g = img[x, y, 1]
                r = img[x, y, 2]
                gray[x, y] = int((int(r) + int(g) + int(b)) / 3)

        # ---------- HISTOGRAM ----------
        hist = np.zeros(256)

        for x in range(0, gray.shape[0]):
            for y in range(0, gray.shape[1]):
                hist[gray[x, y]] += 1

        # ---------- AUTOMATIC THRESHOLD ----------
        peak1 = np.argmax(hist[:128])
        peak2 = np.argmax(hist[128:]) + 128
        threshold = int((peak1 + peak2) / 2)

        # ---------- THRESHOLD ----------
        binary = np.zeros_like(gray)

        for x in range(0, gray.shape[0]):
            for y in range(0, gray.shape[1]):
                if gray[x, y] > threshold:
                    binary[x, y] = 255
                else:
                    binary[x, y] = 0

        # ---------- MORPHOLOGICAL CLOSING ----------
        binary = dilate(binary)
        binary = erode(binary)

        # Save binary result
        cv2.imwrite(os.path.join(output_folder, "binary_" + filename), binary)

        end_time = time.time()
        processing_time = end_time - start_time

        result = "THRESHOLD + CLOSING"

        cv2.putText(img,
                    f"{result} | {processing_time:.4f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        cv2.imwrite(os.path.join(output_folder, filename), img)

        print("Processed:", filename)


if __name__ == "__main__":
    main()