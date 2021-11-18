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
