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


class TDataCatalog:
    def __init__(self):
        self.modules = []
        self.build_catalog()

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
            # cv2.imshow("THIS ITEM IMAGE", comparate.image)
            # # cv2.imshow("CATALOG IMAGE", pocket.image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return pocket  # return item with the best mach (lowest value)

    def get_item(self, name):
        match = ""
        # Not really intended for use besides debugging
        for module in self.modules:
            for item in module.item_list:
                if item.name == name:
                    match = item
                if match:
                    return match


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
    def _read_dims(dim_text: str):  # this really should be done in the TWebScraper, but is here because im lazy
        if dim_text:
            i, j = dim_text.split("x")
            return int(i), int(j)
        return None


if __name__ == "__main__":  # debug purposes, will generate the catalog for testing without doing anything else.

    x = TDataCatalog()
    # an = TItemTypes.TItem("ItemName", 1, (1, 1), False)
    # container = TItemTypes.TContainerItem("ContainerName", 1, (1, 1), False, (5, 5))
    # result = x.compare_to_catalog(container)

    test_image = cv2.imread(r"C:\pyworkspace\tarkovinventoryproject\Data\testcompare\clipped.PNG")
    test_item = TItemTypes.TItem("My_Favorite_Helmet", test_image, (2, 2), False)



    cataloggued = x.get_item("SSh-68_steel_helmet")

    res = test_item.compare_to(cataloggued)
    # cat_im = cataloggued.image
    # h, w, c = cat_im.shape
    # self_resized = cv2.resize(test_item.image, (w, h), interpolation=cv2.INTER_AREA)
    # if c == 4:  # sometimes candidate is BGRA
    #     cat_im = cv2.cvtColor(cat_im, cv2.COLOR_BGRA2BGR)
    #
    # image_sub = cv2.absdiff(self_resized,cat_im)

    print()
    result = x.compare_to_catalog(test_item)
    print()

