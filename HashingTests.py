import numpy as np
import cv2
import vptree
import imagehash



@staticmethod

def compute_hash(image,hashSize=8):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize the grayscale image, adding a single column (width) so we can compute the horizontal gradient
    resized = cv2.resize(gray, (hashSize + 1, hashSize+1))

    # compute the (relative) horizontal gradient between adjacent column pixels
    diff_h = resized[:, 1:] > resized[:, :-1]

    # Test, compute vertical gradient between adjacent row pixels
    diff_v = resized[1:, :] > resized[:-1, :]

    h_sum = sum([2 ** i for (i, v) in enumerate(diff_h.flatten()) if v])
    v_sum = sum([2 ** i for (i, v) in enumerate(diff_h.flatten()) if v])

    joined = int(str(h_sum) + str(v_sum))

    # convert the difference image to a hash
    return joined


def convert_hash(hash):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(hash, dtype="float64"))


def hamming(a, b):
    # compute and return the Hamming distance between the integers
    return bin(int(a) ^ int(b)).count("1")


def hash_image(image):
    hash_dict = {}   # hash : image
    im_hash = compute_hash(image)
    im_hash = convert_hash(im_hash)


def build_vptree():
    hash_dict = {}
    points = list(hash_dict.keys())
    tree = vptree.VPTree(points, hamming)


def search_vptree(image):

    query_hash = hash_image(image)

    my_dict = {12312: "wow"}
    tree = vptree.VPTree(my_dict, hamming)

    results = tree.get_all_in_range(query_hash, max_distance= 10)
    results = sorted(results)

    for (d, h) in results:
        # grab all image paths in our dataset with the same hash
        resultPaths = my_dict.get(h, [])
        print("[INFO] {} total image(s) with d: {}, h: {}".format(
            len(resultPaths), d, h))

        # loop over the result paths
        for resultPath in resultPaths:
            # load the result image and display it to our screen
            result = cv2.imread(resultPath)
            cv2.imshow("Result", result)
            cv2.waitKey(0)





