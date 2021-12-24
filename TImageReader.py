"""
Processes the screenshot(s) into detected TItems.
Contains logic for processing a singular screenshot into TItems.
Will determine if screenshot is of stash or of a container object.
Read items into a dynamic placeholder TContainerItem.
The contents of this container can then be copied to its rightful container object (be it the Stash or a container)
later. (and probably in the TStashManager module)
"""

import TItemTypes
import cv2
import numpy as np
from TCoordinate import TCoordinate


class TImageReader:
    def __init__(self):
        self.current_image = None
        self.parse_image()

    def parse_image(self):  # reads a screenshot, detects any open containers and reads it into appropriate objects
        img_path = r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testimage2.png"
        my_image = cv2.imread(img_path)

        self.current_image = my_image

        containers = self.detect_open_containers(my_image)
        if containers:
            for container in containers:
                self.read_container_image()
        else:
            self.read_stash_image()

        self.parse_items()

    def parse_items(self):
        pass

    @staticmethod
    def detect_open_containers(screenshot) -> list:
        img_hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 1])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)
        invert_mask = cv2.bitwise_not(mask)

        kernal = np.ones((2,2), np.uint8)
        eroded_mask = cv2.erode(invert_mask, kernal, iterations=1)
        dilation_mask = cv2.dilate(eroded_mask, kernal, iterations=1)

        cv2.imshow("opening", dilation_mask)
        img_canny = cv2.Canny(invert_mask, 200, 255)  # edge detect
        contours, hierarchy = cv2.findContours(dilation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        window_locations = []
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 15000:
                cv2.drawContours(screenshot, cont, -1, (0, 255, 0), 1)
                peri = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, .2 * peri, True)

                lower_window_coords = TCoordinate(approx[0][0][0], approx[0][0][1])
                upper_window_coords = TCoordinate(approx[1][0][0], approx[1][0][1])
                window_locations.append((lower_window_coords, upper_window_coords))  # append tuple of low/upper cords

        # cv2.imshow("image", screenshot)

    def read_container_image(self):
        # read a container image into a TContainerItem
        #
        pass

    def read_stash_image(self):
        # read a stash image into a TContainerItem
        pass




if __name__ == "__main__":
    TImageReader()
