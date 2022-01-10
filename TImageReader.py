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
                self.read_container_image(container)
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

        # cv2.imshow("opening", dilation_mask)
        img_canny = cv2.Canny(invert_mask, 200, 255)  # edge detect
        contours, hierarchy = cv2.findContours(dilation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        window_locations = []
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 15000:
                # cv2.drawContours(screenshot, cont, -1, (0, 255, 0), 1)
                peri = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, .2 * peri, True)

                lower_window_coords = TCoordinate(approx[0][0][0], approx[0][0][1])
                upper_window_coords = TCoordinate(approx[1][0][0], approx[1][0][1])
                window_locations.append((lower_window_coords, upper_window_coords))  # append tuple of low/upper cords\

        return window_locations

    def read_container_image(self, coordinate_tuple: tuple):
        # read a container image into TContainerItems

        lower_coords = coordinate_tuple[0]
        upper_coords = coordinate_tuple[1]
        container_image = self.current_image[lower_coords.y:upper_coords.y, lower_coords.x:upper_coords.x]
        header, body = self.container_split__body_and_header(container_image)

    @staticmethod
    def container_split__body_and_header(container_image: np.ndarray) -> "tuple[np.ndarray,np.ndarray]":

        img_hsv = cv2.cvtColor(container_image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 1])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)
        invert_mask = cv2.bitwise_not(mask)

        # Removes some noise
        kernal = np.ones((2, 2), np.uint8)
        eroded_mask = cv2.erode(invert_mask, kernal, iterations=1)
        dilation_mask = cv2.dilate(eroded_mask, kernal, iterations=1)
        cv2.imshow("opening", dilation_mask)
        img_canny = cv2.Canny(invert_mask, 200, 255)  # edge detect
        contours, hierarchy = cv2.findContours(dilation_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Detect Split between header and body.
        lower_window_coords = None
        upper_window_coords = None
        for cont in contours:
            area = cv2.contourArea(cont)
            if 100 < area < 15000:  # only the header section is small enough to be selected
                # cv2.drawContours(container_image, cont, -1, (0, 255, 0), 1)
                peri = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, .2 * peri, True)

                lower_window_coords = TCoordinate(approx[0][0][0], approx[0][0][1])
                upper_window_coords = TCoordinate(approx[1][0][0], approx[1][0][1])

                ####
                header_image = container_image[lower_window_coords.y:upper_window_coords.y,
                               lower_window_coords.x:upper_window_coords.x]

                container_body_image = container_image[upper_window_coords.y:, :]

                cv2.imshow("header", header_image)
                cv2.imshow("containerbody", container_body_image)
                if cv2.waitKey(0) & 0xFF == ord('t'):
                    cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testheader.png", header_image)
                    cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testcontainerbody.png",
                                container_body_image)

                return header_image, container_body_image

        if not lower_window_coords:
            print("ERROR: Container header not found.")
            return None

    def container_body_get_item_outlines(self):
        pass

    def read_stash_image(self):
        # read a stash image into a TContainerItem
        pass


if __name__ == "__main__":
    TImageReader()
