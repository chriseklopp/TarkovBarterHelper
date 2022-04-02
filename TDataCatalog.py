"""
Singleton
This class will manage the catalog of items and their properties
Will initialize from a file containing the necessary information.


# TODO: Switch over to using a SQLite database instead of dataframe.
"""
import os
import cv2
import numpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import TItemTypes
from typing import Union
import vptree

import imagehash
import PIL


class TDataCatalog:
    def __init__(self):
        self.modules = []
        self.build_catalog()

        # Hash Stuff
        self.hash_dict = {}
        self.fill_hash_dict()
        self.VP_tree = self.build_vptree()

    def build_catalog(self):  # builds data catalog from modules in catalog directory
        wd = os.getcwd()
        catalog_directory = os.path.join(wd, "Data", "catalog")
        os.chdir(catalog_directory)
        modules = os.listdir(catalog_directory)
        if not modules:
            print(f"0 modules detected in {catalog_directory}")
            return

        print(f"{len(modules)} modules detected")
        for module in modules:

            print(f"Importing Module: {module}")
            module_path = os.path.join(os.getcwd(), module)
            if not os.path.isdir(module_path):
                print("ERROR: Invalid module: Not a directory")
                continue

            data_module = DataModule(module_path)
            if data_module:
                self.modules.append(data_module)

    # noinspection SpellCheckingInspection
    def compare_to_catalog(self, comparate: TItemTypes.TItem) -> Union[TItemTypes.TItem, None]:
        pocket = None
        pocket_val = 1
        for module in self.modules:
            for item in module.item_list:
                compare_result = comparate.compare_to(item)
                if compare_result:
                    com_item, val = compare_result
                    if val < pocket_val:
                        pocket_val = val
                        pocket = com_item

        if pocket:
            print(f"CATALOG: {pocket.name}, VAL: {pocket_val}")
            ## DEBUG
            # cv2.imshow("THIS ITEM IMAGE", comparate.image)
            # cv2.imshow("CATALOG IMAGE", pocket.image)
            # if cv2.waitKey(0) & 0xFF == ord('t'):
            #     cv2.imwrite(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\imageofinterest.png",
            #                 comparate.image)
            # cv2.destroyAllWindows()
            ## DEBUG
        else:
            print("UNKNOWN IMAGE")
            cv2.imshow("CATALOG IMAGE", comparate.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pocket  # return item with the best mach (lowest value)

    def get_item(self, name):
        match = ""
        # Return matching item matching name from catalog
        for module in self.modules:
            for item in module.item_list:
                if item.name == name:
                    match = item
                if match:
                    return match

    def dump_catalog(self):
        # DEBUG purposes ONLY
        item_list = []
        for module in self.modules:
            for item in module.item_list:
                item_list.append(item)
        return item_list

    def fill_hash_dict(self):
        print("Converting to Hashes")
        item_list = self.dump_catalog()
        for item in item_list:
            # item = item   #type: TItemTypes.TItem
            image_hash = self.hash_image(item.image)

            entry = self.hash_dict.get(image_hash, [])  # access key, if DNE set equal to []
            entry.append(item)
            self.hash_dict[image_hash] = entry
        print("Done Converting to Hashes!")

    @staticmethod
    def compute_hash(image, hash_size=8):
        # convert the image to grayscale

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # resize the grayscale image, adding a single column (width) so we can compute the horizontal gradient
        resized = cv2.resize(gray, (hash_size + 1, hash_size))

        # compute the (relative) horizontal gradient between adjacent column pixels
        diff = resized[:, 1:] > resized[:, :-1]

        # convert the difference image to a hash
        x = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

        return x

    # @staticmethod
    # def compute_hash(image, hashSize=8):
    #     # convert the image to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     # resize the grayscale image, adding a single column (width) so we can compute the horizontal gradient
    #     resized = cv2.resize(gray, (hashSize + 1, hashSize + 1))
    #
    #     # compute the (relative) horizontal gradient between adjacent column pixels
    #     diff_h = resized[:, 1:] > resized[:, :-1]
    #
    #     # Test, compute vertical gradient between adjacent row pixels
    #     diff_v = resized[1:, :] > resized[:-1, :]
    #
    #     h_sum = sum([2 ** i for (i, v) in enumerate(diff_h.flatten()) if v])
    #     v_sum = sum([2 ** i for (i, v) in enumerate(diff_v.flatten()) if v])
    #
    #     joined = int(str(h_sum) + str(v_sum))
    #
    #     # convert the difference image to a hash
    #     return joined

    def hash_image(self, image):
        # hash_dict = {}  # hash : image
        # im_hash = self.compute_hash(image)
        # im_hash = self.convert_hash(im_hash)

        # Testing
        im_pil = PIL.Image.fromarray(image)
        im_hash = str(imagehash.phash(im_pil))
        # awdaw= imagehash.hex_to_hash(im_hash)
        return im_hash

    @staticmethod
    def convert_hash(im_hash):
        # convert the hash to NumPy's 64-bit float and then back to
        # Python's built in int
        return int(np.array(im_hash, dtype="float64"))

    @staticmethod
    def hamming(a, b):
        # compute and return the Hamming distance between the integers

        # return bin(int(a) ^ int(b)).count("1")
        return imagehash.hex_to_hash(a)-imagehash.hex_to_hash(b)



    def build_vptree(self):
        points = list(self.hash_dict.keys())
        tree = vptree.VPTree(points, self.hamming)
        return tree

    def search_vptree(self, image):

        query_hash = self.hash_image(image)
        results = self.VP_tree.get_n_nearest_neighbors(query_hash,5)
        results = sorted(results)

        if not len(results):
            print("0 results in range")

        for (d, h) in results:
            # grab all image paths in our dataset with the same hash
            result_items = self.hash_dict.get(h, [])
            print("[INFO] {} total image(s) with d: {}, h: {}".format(len(result_items), d, h))

            # loop over the result paths
            for item in result_items:
                # load the result image and display it to our screen
                cv2.imshow("Result", item.image)
                cv2.waitKey(0)

@dataclass
class DataModule:

    def __init__(self, module_path):
        self.item_list = []
        self._create_module(module_path)

    def _create_module(self, module_path: str):  # creates module

        module_files = os.listdir(module_path)
        image_directory = os.path.join(module_path, "images")
        if not os.path.exists(image_directory) and os.path.isdir(image_directory):
            print(f"ERROR: No image directory {image_directory}")
            image_directory = None

        for file in module_files:
            file_path = os.path.join(module_path, file)
            if not os.path.isfile(file_path):
                continue

            df = pd.read_csv(file_path)
            df.apply(self._create_items, axis=1)
            # self.read_table(file_path)

    def _create_items(self, df_row):
        index_names = df_row.index
        image_path = df_row["Image_path"]

        if type(image_path) is str:

            stream = open(image_path, 'rb')
            bytes = bytearray(stream.read())
            numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

            npin = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(npin, cv2.IMREAD_UNCHANGED)

            # image = cv2.imread(image_path)

        else:
            image = np.ones((64, 64, 3), np.uint8)  # used as default if image dir or specific image not found
            print(f'{df_row["Name"]} IMAGE NOT FOUND.')
        if type(image) is not np.ndarray:
            image = np.ones((64, 64, 3), np.uint8)  # used as default if image dir or specific image not found
            print(f'{df_row["Name"]} IMAGE NOT FOUND.')

        if "Inner_dims" in index_names:  # this means its some sort of container type object.
            # TODO: will have to support irregular container object, when I get around to implementing that
            try:
                inner_dims = self._read_dims(df_row["Inner_dims"])
                outer_dims = self._read_dims(df_row["Outer_dims"])
                if not outer_dims or not inner_dims:
                    return
                item = TItemTypes.TContainerItem(name=df_row["Name"], image=image, dim=outer_dims,
                                                 rotated=False, container_dims=inner_dims)
                self.item_list.append(item)

            except Exception as e:  # I don't care what this catches, I just don't want it in my catalog
                print("ERROR: Item creation error")
                print(e)
                return

        else:  # this means it is a normal non container object
            try:
                outer_dims = self._read_dims(df_row["Outer_dims"])
                if not outer_dims:
                    return
                item = TItemTypes.TItem(name=df_row["Name"], image=image, dim=outer_dims, rotated=False)
                self.item_list.append(item)

            except Exception as e:  # I don't care what this catches, I just don't want it in my catalog
                print("ERROR: Item creation error")
                print(e)

                return

    @staticmethod
    def _read_dims(dim_text: str):
        if dim_text:
            i, j = dim_text.split("x")
            return int(i), int(j)
        return None


if __name__ == "__main__":  # debug purposes, will generate the catalog for testing without doing anything else.

    x = TDataCatalog()
    # an = TItemTypes.TItem("ItemName", 1, (1, 1), False)
    # container = TItemTypes.TContainerItem("ContainerName", 1, (1, 1), False, (5, 5))
    # result = x.compare_to_catalog(container)

    imageofinterest = cv2.imread(r"C:\pyworkspace\tarkovinventoryproject\Data\screenshots\testitem6.png")
    test_item = TItemTypes.TItem("My_Favorite_Helmet", imageofinterest, (4, 4), False)

    x.search_vptree(test_item.image)
    x.compare_to_catalog(test_item)
    # awdaw = x.get_item("SAS_drive")


    print()

