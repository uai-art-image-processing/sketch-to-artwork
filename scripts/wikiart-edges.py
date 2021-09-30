# from wikiart import CannyThreshold
import os
import argparse
import cv2
import numpy as np
# import matplotlib.pyplot as plt

def CannyThreshold(filename, low_threshold=75, upper_threshold=250):
    # img = cv2.imread(filename)
    # Works with unicode image paths
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                   cv2.IMREAD_UNCHANGED)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(imgray, (5,5), 0)
    edges = cv2.Canny(img_blur, low_threshold, upper_threshold)
    # ret, thresh = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
    # at last, swap blacks and whites
    return np.where(edges != 0, 0, 255)

def procesDir(src, dst, verbose):
    dst = os.path.join(os.getcwd(), dst)
    src = os.path.join(os.getcwd(), src)

    for dirpath, dirnames, filenames in os.walk(src):
        for idx, filename in enumerate(filenames):
            fullpath = os.path.join(dirpath, filename)
            if verbose: print("Processing " + filename, end= "...")
            
            try:
                # Try applying edge detection algo
                edge = CannyThreshold(fullpath)
                # Write new image
                cv2.imwrite(os.path.join(dst, filename), edge)
            except:
                if verbose: print("Failed to process image")
                break
            
            if not verbose: print("\r", f"Processed {(idx+1)/len(filenames)*100:.1f}%", end="", sep="")
            if verbose: print("Succeded")
            
            
    print("\nDone")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source")
    parser.add_argument("-d", "--destiny")
    parser.add_argument("-v", "--verbose")
    args = parser.parse_args()

    if args.source is not None and args.destiny is not None:
        print("Processing images...")
        procesDir(args.source, args.destiny, args.verbose)
    else:
        raise Exception("Not enough arguments")

    