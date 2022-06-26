"""
Contains my homegrown character detection algorithms to find text in images.
Kind of sucks.

Also includes code for saving a training dataset of characters to be used for ML purposes.


"""



from TCoordinate import TCoordinate
import numpy as np
import cv2
import math
import TDataCatalog


class pixel_object:
    def __init__(self, image, first_point):

        self.low_coords = (TCoordinate(5000, 5000))
        self.upper_coords = TCoordinate(0, 0)
        pixel_list = [first_point]
        self.find_neighbors(image, first_point, pixel_list)
        self.valid = False
        for pixel in pixel_list:
            if pixel[0] < self.low_coords.x:
                self.low_coords.x = pixel[0]

            if pixel[1] < self.low_coords.y:
                self.low_coords.y = pixel[1]

            if pixel[0] > self.upper_coords.x:
                self.upper_coords.x = pixel[0]

            if pixel[1] > self.upper_coords.y:
                self.upper_coords.y = pixel[1]

        if (self.low_coords.x < self.upper_coords.x) & (self.low_coords.y < self.upper_coords.y):
            self.valid = True
        self.area = (self.upper_coords.x - self.low_coords.x) * (self.upper_coords.y - self.low_coords.y)

    def find_neighbors(self, image, pixel, pixel_list):
        region = [-1, 0, 1]
        neighbors = []
        range_x = [pixel[0] + i for i in region]
        range_y = [pixel[1] + i for i in region]
        h, w = image.shape

        for x in range_x:
            for y in range_y:
                if (x == h) or (y == w):
                    continue
                if (image[x, y] == 255) & ((x, y) not in pixel_list):
                    pixel_list.append((x, y))
                    neighbors.append((x, y))

        if neighbors:
            for pixel in neighbors:
                self.find_neighbors(image, pixel, pixel_list)


def find_isolated_binary_object(binary_img):
    # takes a binary image. finds bounds of isolated objects, assumes that a box can be drawn around the extrema
    # of the object with NO OTHER objects in that region.
    detected_objects = []
    h, w = binary_img.shape
    detected_map = np.zeros((h, w))
    current_pixel = (0, 0)
    for x in range(1, h-1):
        for y in range(1, w-1):
            if (binary_img[x, y] == 255) & (detected_map[x, y] == 0):
                candidate_object = pixel_object(binary_img, (x, y))
                if candidate_object.valid:
                    detected_map[candidate_object.low_coords.x:candidate_object.upper_coords.x+1,
                                 candidate_object.low_coords.y:candidate_object.upper_coords.y+1] = 1
                    detected_objects.append(candidate_object)

    # for this_object in detected_objects:
    #     test_image = binary_img[this_object.low_coords.x-1:this_object.upper_coords.x+1,
    #                              this_object.low_coords.y-1:this_object.upper_coords.y+1]
    #     cv2.imshow("testimage", test_image)
    #     cv2.waitKey(0)
    # print()
    return detected_objects


def celltextfind(img, cont=False, blob=False):
    h, w, c = img.shape

    img_top = img[:15, :]

    img_gray = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
    mean = img_gray[img_gray > 79].mean()
    col_stdev = np.std(img_gray, axis=0)
    avg_std = math.floor(np.mean(col_stdev))

    null_col_indices = list(col_stdev < avg_std)
    for i, item in enumerate(null_col_indices):
        if item:
            img_gray[:,i] = 0

    lower_thresh = cv2.threshold(img_gray, .86 * mean, 255, cv2.THRESH_BINARY)[1]
    upper_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)[1]

    detected_contours = find_isolated_binary_object(lower_thresh)

    for cont in detected_contours:
        crop = img[cont.low_coords.x-1:cont.upper_coords.x+1,
                   cont.low_coords.y-1:cont.upper_coords.y+1]

        resized = cv2.resize(crop, (100,100), interpolation=cv2.INTER_AREA)
        cv2.imshow("cropped", crop)
        cv2.imshow("resized", resized)
        cv2.imshow("img", img)
        cv2.waitKey(0)

def make_header_training_set():

    pass


if __name__ == "__main__":

    print()

