"""
Base Class for all inventory items.
Most items will probably just be of this type, however a few potential sub class extensions could be useful, such as
weapons, bags, and cases.
"""
from dataclasses import dataclass
import numpy as np


# On start up, TDataCatalog will create an instance of every class.
# When an item is detected and created, in __init__ it will be compared to the catalog in name, image, dimensions
# If a match is found the candidate will have its attributes set to the values of the catalog's.

@dataclass(frozen=True)
class TItem:

    name: str
    item_type: str
    image: np.ndarray
    dim: tuple
    rotated: False


@dataclass(frozen=True)
class TContainerItem(TItem):
    container_dims: tuple

    def __post_init__(self):
        pass