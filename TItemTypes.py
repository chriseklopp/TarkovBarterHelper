"""
Base Class for all inventory items.
Most items will probably just be of this type, however a few potential sub class extensions could be useful, such as
weapons, bags, and cases.

"""
from dataclasses import dataclass, field
import numpy as np


# On start up, TDataCatalog will create an instance of every class.
# When an item is detected and created, in __init__ it will be compared to the catalog in name, image, dimensions
# If a match is found the candidate will have its attributes set to the values of the catalog's.

@dataclass
class TItem:

    name: str
    image: np.ndarray
    dim: tuple  # (rows,columns)
    rotated: bool


@dataclass
class TContainerItem(TItem):
    container_dims: tuple
    container_contents: list = field(init=False)  # list of (TItem, top_left_square_location)
    container_grid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.container_grid = np.zeros(self.container_dims)
        self.container_contents = []

    def insert_item(self):  # insert an item at a location into the container
        pass

    def list_contents(self):  # list of all items and their locations in the container
        pass

    def display_contents(self):  # visually display item contents of a container (possibly filled out in the future)
        pass


@dataclass
class TStash(TContainerItem):
    """
    Stash Sizes: (10x28) (10x38) (10x48) (10x68)
    """
    name: str = "Stash"
    image: np.ndarray = np.zeros((1, 1))
    dim: tuple = (1, 1)
    rotated: bool = False
    container_dims: tuple = (2, 5)

    def change_dims(self, new_dims):
        self.dim = new_dims
        self._resize_grid(new_dims)

    def _resize_grid(self, new_dims: tuple, bottom=True):
        pass


class ItemCreator:  # creates the actual items by comparing them against the items in the DataCatalog.
    def __init__(self):
        pass


if __name__ == "__main__":  # debug purposes
    item = TItem("ItemName", 1, (1, 1), False)
    print(item)
    container = TContainerItem("ContainerName", 1, (1,1),False, (5,5))
    print(container)
    stash = TStash()
    print(stash)
    print()


