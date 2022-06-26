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
import TDataCatalog
import TTextDetection
import os
import datetime
import pytesseract


# TODO: Rewrite / Clean Up this ugly mess

class TImageReader:
    def __init__(self):
        self.current_image = None
        self.cell_size = 84  # NEED TO FIND A WAY TO RELIABLY CALCULATE THIS W/O HARD CODING IT.
        self.container_list = []
        self.header_images = []

    def run(self, image_path):
        self.container_list = []  # reset this in case the reader is run again.
        self.parse_image(image_path)
        self.save_headers()

    def parse_image(self, image_path):  # reads a screenshot, detects any open containers and reads it into appropriate objects

        my_image = cv2.imread(image_path)

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

        kernal = np.ones((2, 2), np.uint8)
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
        if (body is None) or (header is None):
            return

        # create empty container. Can put crap in here when needed.
        name = self.get_container_name(header)
        self.header_images.append(header)  # allows headers to be saved and extracted from the class for ML training
        container = TItemTypes.TContainerItem("PLACEHOLDER", None, (5, 5), False, (20, 20))
        item_outlines = self.get_item_outlines(body)
        self.read_item_outlines(body, item_outlines, container)
        self.container_list.append(container)

    @staticmethod
    def container_split__body_and_header(container_image: np.ndarray) -> "tuple[np.ndarray,np.ndarray]":
        img_gray = cv2.cvtColor(container_image,cv2.COLOR_BGRA2GRAY)

        img_hsv = cv2.cvtColor(container_image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 1])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)
        invert_mask = cv2.bitwise_not(mask)

        # Removes some noise
        kernal = np.ones((2, 2), np.uint8)
        eroded_mask = cv2.erode(invert_mask, kernal, iterations=1)
        dilation_mask = cv2.dilate(eroded_mask, kernal, iterations=1)
        # cv2.imshow("opening", dilation_mask)
        img_canny = cv2.Canny(invert_mask, 200, 255)  # edge detect
        contours, hierarchy = cv2.findContours(dilation_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Detect Split between header and body.
        image_area = container_image.shape[0] * container_image.shape[1]
        lower_window_coords = None
        upper_window_coords = None

        for cont in contours:
            area = cv2.contourArea(cont)
            if 100 < area < image_area * .2:  # only the header section is small enough to be selected
                # cv2.drawContours(container_image, cont, -1, (0, 255, 0), 1)
                peri = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, .2 * peri, True)

                lower_window_coords = TCoordinate(approx[0][0][0], approx[0][0][1])
                upper_window_coords = TCoordinate(approx[1][0][0], approx[1][0][1])

                header_image = container_image[lower_window_coords.y:upper_window_coords.y,
                                               lower_window_coords.x:upper_window_coords.x]

                container_body_image = container_image[upper_window_coords.y:, :]

                # cv2.imshow("header", header_image)
                # cv2.imshow("containerbody", container_body_image)

                # DEBUG

                #     cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testcontainerbody.png",
                #                 container_body_image)
                head_gray_debug = cv2.cvtColor(header_image,cv2.COLOR_BGRA2GRAY)
                return header_image, container_body_image

        if not lower_window_coords:
            print("ERROR: Container header not found.")
            return None, None

    def read_item_outlines(self, source_image, item_locations, container: TItemTypes.TContainerItem):
        # acts inplace on the container suppliedgggg
        for item in item_locations:
            lower_coord, upper_coord = item
            print()
            item_cropped = source_image[lower_coord.y:upper_coord.y, lower_coord.x:upper_coord.x]  # This is the image.

            # calculate cell location.
            x, y = lower_coord.values()
            cells_right = round(x/self.cell_size)
            cells_down = round(y/self.cell_size)

            # calculate item dimensions.
            x, y = upper_coord.values()
            lower_corner_cells_right = round(x / self.cell_size)
            lower_corner_cells_down = round(y / self.cell_size)
            dim_x = lower_corner_cells_right - cells_right
            dim_y = lower_corner_cells_down - cells_down

            # make item
            this_item = TItemTypes.TItem("Unknown", item_cropped, (dim_x, dim_y), False)


            # # Testing of Hashing Algo
            # cv2.imshow("THE ORIGINAL", this_item.image)
            # cv2.waitKey(0)
            # DataCatalog.search_vptree(this_item.image)
            match_item = DataCatalog.hash_template_match(this_item)

            if match_item is None:
                pass # this would happen if NONE of the offerred items have the correct dims in com[are_to
            else:
                this_item = match_item
            #
            # Compare it to the Catalog.
            # compared_item = DataCatalog.compare_to_catalog(this_item)
            # if compared_item is not None:
            #     this_item = compared_item  # This happens when there was no match.

            container.insert_item(this_item, (cells_right, cells_down))



        return

    def get_item_outlines(self, body_image: np.ndarray) -> "list[tuple[TCoordinate,TCoordinate]]":
        # Identify items in body, return list of their locations.

        image_area = body_image.shape[0] * body_image.shape[1]

        img_gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)
        lower_thresh = cv2.threshold(img_gray, 78, 255, cv2.THRESH_BINARY)[1]
        upper_thresh = cv2.threshold(img_gray, 79, 255, cv2.THRESH_BINARY)[1]
        combined_thresh = cv2.bitwise_xor(lower_thresh, upper_thresh)

        contours, hierarchy = cv2.findContours(combined_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        coord_pairs = []
        for cont in contours:
            area = cv2.contourArea(cont)
            # < .95 * image_area ensures that if the WHOLE body is identified it wont be counted.
            min_size = (self.cell_size ** 2) * .85
            if min_size < area < .90 * image_area:
                peri = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, .2 * peri, True)

                d,h,w = approx.shape
                if d != 2:
                    continue

                lower_grid_coords = TCoordinate(approx[0][0][0] + 1,
                                                approx[0][0][1])  # +1 is for a line detection correction
                upper_grid_coords = TCoordinate(approx[1][0][0], approx[1][0][1])

                if upper_grid_coords.x < lower_grid_coords.x:  # I have no idea why this is necessary
                    upp = upper_grid_coords.x
                    low = lower_grid_coords.x
                    upper_grid_coords.x = low
                    lower_grid_coords.x = upp

                if upper_grid_coords.y < lower_grid_coords.y:  # I have no idea why this is necessary
                    upp = upper_grid_coords.y
                    low = lower_grid_coords.y
                    upper_grid_coords.y = low
                    lower_grid_coords.y = upp

                coord_pairs.append((lower_grid_coords, upper_grid_coords))

                item_cropped = body_image[lower_grid_coords.y:upper_grid_coords.y,
                                          lower_grid_coords.x:upper_grid_coords.x]

                # print(lower_grid_coords.values())
                # print(upper_grid_coords.values())
                # # cv2.drawContours(img_copy, cont, -1, (0, 255, 0), 1)
                # cv2.imshow("this contour", item_cropped)
                # # cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\THISITEM.png",
                # #             item_cropped)
                # # cv2.imshow("img", body_image)
                # cv2.waitKey(0)

        return coord_pairs

    def get_container_name(self, header):
        name = "PLACEHOLDER"

        h, w, c = header.shape
        header = header[:, :round(w/2)]
        header_HSV = cv2.cvtColor(header, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 7, 0])
        upper = np.array([179, 255, 255])
        mask = cv2.inRange(header_HSV, lower, upper)
        invert_mask = cv2.bitwise_not(mask)
        characters = TTextDetection.find_isolated_binary_object(invert_mask)

        return name

    def save_headers(self):
        # saves header to file for use in ml training
        wd = os.getcwd()
        raw_header_dir = os.path.join(wd, "Data", "training", "headers", "raw")
        for header in self.header_images:
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            cv2.imwrite(f"{raw_header_dir}/header_{suffix}.png", header)


    def read_stash_image(self):
        # read a stash image into a TContainerItem
        pass


if __name__ == "__main__":
    global DataCatalog  # type: TDataCatalog.TDataCatalog
    DataCatalog = TDataCatalog.TDataCatalog()
    reader = TImageReader()

    wd = os.getcwd()
    raw_header_dir = os.path.join(wd, "Data", "screenshots", "raw")
    images = os.listdir(raw_header_dir)
    for image in images:
        image_path = os.path.join(raw_header_dir, image)
        reader.run(image_path)
    print()


    # for container in reader.container_list:
    #     container.enumerate_contents()

    print()
