import cv2
import sys
import numpy as np

def mean2(matrix):
    sum = 0
    count = 0
    for row in matrix:
        for element in row:
            sum += element
            count += 1
    mean_value = sum / count
    return mean_value

def main(name):
    img = cv2.imread(name, 0)
    print(f"NumPy: {np.mean(img)}")
    print(f"mean2: {mean2(img)}")

if __name__ == "__main__":
    main(sys.argv[1])
    #main("./img.png")