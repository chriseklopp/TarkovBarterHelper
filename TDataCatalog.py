"""
Singleton
This class will manage the catalog of items and their properties
Will initialize from a file containing the necessary information.

Items are hashed using perceptual hash on their image and placed into a dictionary.
This dictionary is copied into a Vp Tree for fast image comparisons.

"""
import os
import cv2
import numpy
import numpy as np
import pandas as pd
import TItemTypes
from typing import Union
import vptree

import imagehash
import PIL


class TDataCatalog:
    def __init__(self):
        self.hash_dict = {}
        self.VP_tree = None
        self._build_catalog()
        self.VP_tree = self._build_vptree()

    def hash_template_match(self, item, num_neighbors=5, compare_threshold=1):
        # Current Best Item Matching Algorithm.
        # PROCESS:
        # 1) hash image
        # 2) Find n nearest neighbors
        # 3) Perform TItem compare_to operation on the n neighbors (includes template matching and dim matching)
        # 4) Return best match
        neighbors = self.search_vptree(item.image, num_neighbors=num_neighbors)
        best_match = self.get_best_catalog_match(item, compare_threshold, subset=neighbors)
        return best_match

    def get_best_catalog_match(self, item, threshold=1, subset=None):
        # Find best catalog match to the provided item, using the compare_to method of TItems.
        # subset is a list specifying the subset of catalog items to compare to.
        # If subset = None (default) will compare provided item to ALL catalog items.
        # Only results with an image difference < threshold will be accepted (default 1, will accept all)
        best_match = None
        best_match_value = 1
        if subset:
            # return best item out of the provided subset
            for cat_item in subset:
                compare_result = item.compare_to(cat_item)
                if compare_result:
                    com_item, val = compare_result
                    if (val < best_match_value) and (val < threshold):
                        best_match_value = val
                        best_match = com_item
            return best_match

        else:
            # return best item out of the whole catalog
            for entry_list in self.hash_dict.values():
                for cat_item in entry_list:
                    compare_result = item.compare_to(cat_item)
                    if compare_result:
                        com_item, val = compare_result
                        if (val < best_match_value) and (val < threshold):
                            best_match_value = val
                            best_match = com_item
            return best_match

    def get_item(self, name):
        # Return matching item matching name from catalog
        match = ""
        for entry_list in self.hash_dict.values():
            for item in entry_list:
                if item.name == name:
                    match = item
                if match:
                    return match
        print(f"{name} not found in catalog")
        return None

    def dump_catalog(self):
        # Mainly for DEBUG
        item_list = []
        for entry_list in self.hash_dict.values():
            for cat_item in entry_list:
                item_list.append(cat_item)
        return item_list

    def search_vptree(self, image, num_neighbors=6):
        # search the vp tree for the n nearest neighbors to the provided image and return them
        query_hash = self._hash_image(image)
        results = self.VP_tree.get_n_nearest_neighbors(query_hash, num_neighbors)
        results = sorted(results)

        result_list = []
        for (d, h) in results:

            result_items = self.hash_dict.get(h, [])
            result_list += result_items
            print("[INFO] {} total image(s) with d: {}, h: {}".format(len(result_items), d, h))

            # # loop over the result images. DEBUG
            # for item in result_items:
            #     cv2.imshow("Result", item.image)
            #     cv2.waitKey(0)

        return result_list

    def _build_catalog(self):  # builds data catalog from modules in catalog directory
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

            self._read_module(module_path)
        os.chdir(wd)  # reset wd to og value

    def _build_vptree(self):
        # build the vp_tree from the hash_dict
        points = list(self.hash_dict.keys())
        tree = vptree.VPTree(points, self._hamming)
        return tree

    def _read_module(self, module_path):
        # read a catalog module (sub-folder) into TItems, and pass them to the hash dictionary.
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
            df.apply(self._process_items, axis=1)
            # self.read_table(file_path)

    def _process_items(self, df_row):

        index_names = df_row.index
        image_path = df_row["Image_path"]

        if type(image_path) is str:

            # stream = open(image_path, 'rb')
            # bytes = bytearray(stream.read())
            # numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
            # image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            #
            npin = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(npin, cv2.IMREAD_UNCHANGED)

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
                self._append_hash_dict(item)

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
                self._append_hash_dict(item)

            except Exception as e:  # I don't care what this catches, I just don't want it in my catalog
                print("ERROR: Item creation error")
                print(e)

                return

    def _append_hash_dict(self, item: TItemTypes.TItem):

        image_hash = self._hash_image(item.image)
        entry = self.hash_dict.get(image_hash, [])  # access key, if DNE set equal to []
        entry.append(item)
        self.hash_dict[image_hash] = entry

    @staticmethod
    def _hash_image(image):
        im_pil = PIL.Image.fromarray(image)
        im_hash = str(imagehash.phash(im_pil))

        return im_hash

    @staticmethod
    def _hamming(a, b):
        # compute and return the Hamming distance between the integers
        return imagehash.hex_to_hash(a)-imagehash.hex_to_hash(b)

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
    awd = x.get_best_catalog_match(test_item)
    # awdaw = x.get_item("SAS_drive")


    print()

