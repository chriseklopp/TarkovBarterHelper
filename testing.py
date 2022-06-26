import os
import shutil
import requests
import cv2
import numpy as np
import math
import time
import TItemTypes
import TDataCatalog
import sys
from TCoordinate import TCoordinate
from PIL import Image
import pytesseract
import TDataCatalog
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def hsvtrackbar(img_path):
    # create window
    cv2.namedWindow("TrackBarWindow")
    cv2.resizeWindow('TrackBarWindow', 640,240)

    #trackbars
    def track(x):
        print(lower,upper)
    cv2.createTrackbar("huemin","TrackBarWindow",0,179,track)
    cv2.createTrackbar("huemax","TrackBarWindow",179,179,track)
    cv2.createTrackbar("satmin","TrackBarWindow",0,255,track)
    cv2.createTrackbar("satmax","TrackBarWindow",255,255,track)
    cv2.createTrackbar("valmin","TrackBarWindow",0,255,track)
    cv2.createTrackbar("valmax","TrackBarWindow",255,255,track)

    while True:
        img = cv2.imread(img_path)


        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("huemin","TrackBarWindow")
        h_max = cv2.getTrackbarPos("huemax","TrackBarWindow")
        s_min = cv2.getTrackbarPos("satmin","TrackBarWindow")
        s_max = cv2.getTrackbarPos("satmax","TrackBarWindow")
        v_min = cv2.getTrackbarPos("valmin","TrackBarWindow")
        v_max = cv2.getTrackbarPos("valmax","TrackBarWindow")
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv2.inRange(imgHSV,lower,upper)
        imgResult = cv2.bitwise_and(img,img,mask=mask)
        cv2.imshow("OG", img)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", imgResult)
        if cv2.waitKey(1) &0xFF == ord('t'):
            cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testmask.png", mask)
            #cv2.imwrite(r"images\hsvtest.png")
            print(mask)
            return (mask)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break


def headertextfind(img, cont=False, blob=False):
    # img = cv2.imread(img_path)
    # kernal = np.ones((2, 2), np.uint8)

    img_hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 7, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    invert_mask = cv2.bitwise_not(mask)

    detected_contours = find_isolated_binary_object(invert_mask)

    for cont in detected_contours:
        crop = img[cont.low_coords.x-1:cont.upper_coords.x+1,
                   cont.low_coords.y-1:cont.upper_coords.y+1]
        cv2.imshow("cropped", crop)
        cv2.waitKey(0)


    # # combined_dilate = cv2.dilate(combined, kernal, iterations = 5)
    # # eroded_mask = cv2.erode(combined_dilate, kernal, iterations=5)
    # # cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\combined.png", combined)
    # # cv2.imshow("combined", combined)
    #
    #
    #
    # eroded_mask = cv2.erode(invert_mask, kernal, iterations=1)
    # kernal = np.ones((3, 3), np.uint8)
    # dilation_mask = cv2.dilate(eroded_mask, kernal, iterations=4)
    # inverted_dilate = cv2.bitwise_not(dilation_mask)
    # # h, w = inverted_dilate.shape
    # # inverted_dilate[h-1:h, :] = 255
    #
    # # cv2.imshow("opening", dilation_mask)
    # # img_canny = cv2.Canny(invert_mask, 50, 200)  # edge detect
    #
    # if blob:
    #     detector = cv2.SimpleBlobDetector_create()
    #     params = cv2.SimpleBlobDetector_Params()
    #     params.filterByArea = False
    #     params.minArea = 10
    #     params.filterByColor = False
    #     params.filterByCircularity = False
    #     params.filterByConvexity = False
    #     params.filterByInertia = False
    #
    #
    #     # Detect blobs.
    #     keypoints = detector.detect(inverted_dilate)
    #     im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
    #                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #
    # if cont:
    #     contours, hierarchy = cv2.findContours(dilation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     for cont in contours:
    #         area = cv2.contourArea(cont)
    #         if 100 < area:
    #             # cv2.drawContours(img, cont, -1, (0, 255, 0), 1)
    #             peri = cv2.arcLength(cont, True)
    #             approx = cv2.approxPolyDP(cont, .1 * peri, True)
    #             x, y, w, h = cv2.boundingRect(cont)
    #
    #     lower_grid_coords = TCoordinate(approx[0][0][0],
    #                                     approx[0][0][1])  # +1 is for a line detection correction
    #     upper_grid_coords = TCoordinate(approx[1][0][0], approx[1][0][1])
    #
    #     if upper_grid_coords.x < lower_grid_coords.x:  # I have no idea why this is necessary
    #         upp = upper_grid_coords.x
    #         low = lower_grid_coords.x
    #         upper_grid_coords.x = low
    #         lower_grid_coords.x = upp
    #
    #     if upper_grid_coords.y < lower_grid_coords.y:  # I have no idea why this is necessary
    #         upp = upper_grid_coords.y
    #         low = lower_grid_coords.y
    #         upper_grid_coords.y = low
    #         lower_grid_coords.y = upp
    #
    #     cropped = img[lower_grid_coords.y:upper_grid_coords.y,
    #                    lower_grid_coords.x:upper_grid_coords.x]
    #
    #     cv2.imshow("cropped", cropped)
    #     cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\textsample.png", cropped)
    #
    #
    # cv2.imshow("invert_mask", invert_mask)
    # cv2.imshow("img", img)
    # cv2.imshow("dilation_mask", dilation_mask)
    # cv2.imshow("inverted_dilate", inverted_dilate)
    # cv2.waitKey(0)


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
    gcopy = img_gray.copy()



    mean = img_gray[img_gray > 79].mean()
    # col_stdev = np.std(img_gray, axis=0)
    # avg_std = math.floor(np.mean(col_stdev))
    #
    # null_col_indices = list(col_stdev < avg_std)
    # for i, item in enumerate(null_col_indices):
    #     if item:
    #         img_gray[:,i] = 0


    # img_gray[img_gray < .86*mean] = 0
    custom_config = r'-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz- --psm 6'
    tess_string = pytesseract.image_to_string(img_gray, config=custom_config)
    print(tess_string)
    cv2.imshow("item", img_gray)
    cv2.waitKey(0)
    return



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



if __name__ == "__main__":
    global DataCatalog  # type: TDataCatalog.TDataCatalog
    DataCatalog = TDataCatalog.TDataCatalog()
    list_of_items = DataCatalog.dump_catalog()
    for item in list_of_items:
        celltextfind(item.image, cont=True, blob=False)
    img_path = r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testheader.png"
    img2 = r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testitem3.png"
    # headertextfind(img_path, cont=True, blob=False)
    celltextfind(img2, cont=True, blob=False)
    # hsvtrackbar(img2)


"""
header text
0,7,0
179,255,255
"""

"""
Item text
0,25,0
179,255,255

"""
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

