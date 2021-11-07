"""
Processes the screenshot(s) into detected TItems.
Contains logic for processing a singular screenshot into an ARRAY of TItems (this MAY be some custom type object)
(This array structure will probably be important for out of game accurate inventory representation, as well
as (most importantly) verification of detection accuracy, avoidance of double counting, and allocation of items to the
correct container.


"""

import cv2


class TImageReader:
    def __init__(self):
        self.parse_image()

    def parse_image(self):
        img_path = r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testimage.png"
        my_image = cv2.imread(img_path)
        cv2.imshow("image", my_image)
        cv2.waitKey(0)
        print()
        self.parse_items()

    def parse_items(self):
        pass


if __name__ == "__main__":
    TImageReader()
