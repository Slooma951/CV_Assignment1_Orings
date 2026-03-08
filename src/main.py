import cv2
import numpy as np
import os
import time


# ---------------- LOAD IMAGES ----------------
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


# ---------------- CONNECTED COMPONENT LABELLING ----------------
def connected_components(binary):

    rows, cols = binary.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    label = 1
    areas = {}

    for i in range(rows):
        for j in range(cols):

            if binary[i, j] == 255 and labels[i, j] == 0:

                stack = [(i, j)]
                area = 0

                while stack:
                    x, y = stack.pop()

                    if (0 <= x < rows and
                        0 <= y < cols and
                        binary[x, y] == 255 and
                        labels[x, y] == 0):

                        labels[x, y] = label
                        area += 1

                        stack.append((x+1, y))
                        stack.append((x-1, y))
                        stack.append((x, y+1))
                        stack.append((x, y-1))

                areas[label] = area
                label += 1

    return labels, areas


# ---------------- COUNT HOLES ----------------
def count_holes(binary):

    rows, cols = binary.shape

    # Invert mask
    inverted = np.zeros_like(binary)

    for i in range(rows):
        for j in range(cols):
            if binary[i, j] == 0:
                inverted[i, j] = 255
            else:
                inverted[i, j] = 0

    labels, areas = connected_components(inverted)

    hole_count = 0

    for label in areas:

        touches_border = False

        # Check if region touches image border (background region)
        for i in range(rows):
            if labels[i, 0] == label or labels[i, cols-1] == label:
                touches_border = True

        for j in range(cols):
            if labels[0, j] == label or labels[rows-1, j] == label:
                touches_border = True

        if not touches_border:
            hole_count += 1

    return hole_count


# ---------------- MAIN ----------------
def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_folder = os.path.join(base_dir, "Orings")
    output_folder = os.path.join(base_dir, "outputs")

    images, filenames = load_images(image_folder)

    for img, filename in zip(images, filenames):

        start_time = time.time()

        # ---------- GRAYSCALE ----------
        gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                b = int(img[x, y, 0])
                g = int(img[x, y, 1])
                r = int(img[x, y, 2])
                gray[x, y] = int((r + g + b) / 3)

        # ---------- HISTOGRAM ----------
        hist = np.zeros(256)

        for x in range(gray.shape[0]):
            for y in range(gray.shape[1]):
                hist[gray[x, y]] += 1

        peak1 = np.argmax(hist[:128])
        peak2 = np.argmax(hist[128:]) + 128
        threshold = int((peak1 + peak2) / 2)

        # ---------- THRESHOLD ----------
        binary = np.zeros_like(gray)

        for x in range(gray.shape[0]):
            for y in range(gray.shape[1]):
                if gray[x, y] < threshold:   # O-ring is darker
                    binary[x, y] = 255
                else:
                    binary[x, y] = 0

        # ---------- MORPHOLOGICAL CLOSING ----------
        binary = dilate(binary)
        binary = erode(binary)

        # ---------- CONNECTED COMPONENTS ----------
        labels, areas = connected_components(binary)

        if len(areas) == 0:
            result = "FAIL"
        else:
            largest_label = max(areas, key=areas.get)

            ring_mask = np.zeros_like(binary)

            for i in range(binary.shape[0]):
                for j in range(binary.shape[1]):
                    if labels[i, j] == largest_label:
                        ring_mask[i, j] = 255

            binary = ring_mask

            # ---------- HOLE ANALYSIS ----------
            holes = count_holes(binary)

            if holes == 1:
                result = "PASS"
            else:
                result = "FAIL"

        # ---------- SAVE OUTPUT ----------
        cv2.imwrite(os.path.join(output_folder, "binary_" + filename), binary)

        end_time = time.time()
        processing_time = end_time - start_time

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