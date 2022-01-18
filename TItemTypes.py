"""
Base Class for all inventory items.
Most items will probably just be of this type, however a few potential sub class extensions could be useful, such as
weapons, bags, and cases.

"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import copy
from typing import Union

# On start up, TDataCatalog will create an instance of every class.
# When an item is detected and created, in __init__ it will be compared to the catalog in name, image, dimensions
# If a match is found the candidate will have its attributes set to the values of the catalog's.


@dataclass
class TItem:

    name: str
    image: np.ndarray
    dim: tuple  # (rows,columns)
    rotated: bool

    def __post_init__(self):
        if self.image is None:
            self.image = np.ones((64, 64, 3), np.uint8)

    def compare_to(self, candidate: "TItem") -> "TItem":
        """
        Compares a candidate TItem from the catalog to this item.
        Intended to be used on an item from an image,with candidate being a catalog item.
        Determine if the candidate has sufficient matching information to the catalog version.
        If so, it will return a copy of the cataloged version (which contains more information)
        with potential modifications. IE: rotated = True
        Returns None if no match (YOU MUST CHECK FOR NONE WHEN USING THIS METHOD)
        """
        rotation = self._compare_dimensions(candidate)
        if rotation == 2:
            return None  # these images cannot be the same as they have different dimensions. (this would break weapons)

        threshold = .3  # this is just an arbitrary value, can be changed as more testing is done.
        template_match = self._match_template(candidate)
        if template_match < threshold:
            print(f"Match: {candidate.name}. Similarity: {template_match}")
            # DEBUG
            cv2.imshow("THIS ITEM IMAGE", self.image)
            cv2.imshow("CATALOG IMAGE", candidate.image)
            cv2.waitKey(0)

            return self._copy_contents(candidate)

        if self._compare_name():
            # return self._copy_contents(candidate)
            pass

        return None  # if no match is found will just return None

    def _match_template(self, candidate: "TItem") -> float:
        """
        Uses template matching to compare two images.
        """

        candidate_image = candidate.image
        h, w, c = candidate_image.shape

        if c == 4:  # sometimes candidate is BGRA
            candidate_image = cv2.cvtColor(candidate_image, cv2.COLOR_BGRA2BGR)

        h, w, c = self.image.shape
        candidate_image = cv2.resize(candidate_image, (w, h), interpolation=cv2.INTER_AREA)
        # want them to be the same size.

        # sqdiff = cv2.matchTemplate(self_resized, candidate_image, cv2.TM_SQDIFF)  # FIX THIS
        sqdiffnorm = cv2.matchTemplate(self.image, candidate_image, cv2.TM_SQDIFF_NORMED)
        return sqdiffnorm

    def _copy_contents(self, candidate: "TItem") -> "TItem":
        """
        Used when a match between candidate and this object self detected.
        :param candidate:
        :return: TItem
        """

        candidate_copy = copy.deepcopy(candidate)  # type: TItem

        rotation = self._detect_rotation(candidate)
        candidate_copy.rotated = rotation

        candidate_copy.image = self.image
        # results in slight differences, such as durability to be kept
        # or in the case of weapons, for their potentially modded image to not be replaced with the stock one.

        self.dim = ()
        return candidate_copy

    def _detect_rotation(self, candidate) -> bool:
        """
        Only really useful after a decision to copy has been initiated, compares dims of both items and
        determines if the dims of self are rotated compared to catalog candidate.

        """
        i_self, j_self = self.dim
        if i_self == j_self:  # if it is a square object rotation doesnt matter.
            return False
        i_can, j_can = candidate.dim
        if i_self == i_can and j_self == j_can:  # this object is not rotated.
            return False

        if i_self == j_can and j_self == i_can:  # dimensions of the object are swapped, so its rotated
            return True

        # final case doesn't need to be stated, but if dimensions are not equal return false
        # can occur as some items change their dimensions from reference dims based on attachments (like suppressors)
        return False

    def _compare_dimensions(self,candidate) -> int:
        """
        0 = Identical
        1 = Rotated
        2 = Non match
        """
        i_self, j_self = self.dim
        i_can, j_can = candidate.dim

        if self.dim == candidate.dim:  # If dimensions are equal. (doesnt guarantee that it is not rotated)
            return 0

        if i_self == j_can and j_self == i_can:  # dimensions are rotated
            return 1

        return 2  # non matchable dimensions

    def _compare_name(self):
        """
        NYI
        Secondary form of matching, if two items have the same name they will be considered a match.
        Useful for stuff like weapons which can be heavily modded to look COMPLETELY different with different dims.
        This is dependant on me being able to visually extract the text from screenshots.

        This is a function, as im not sure to the accuracy a text extraction method would have, nor to the similiarity
        of in-game item names to their technical names from the wiki.
        Will likely have to compare substrings of each to get a useful match.
        """

        name = self.name
        return None


@dataclass
class TContainerItem(TItem):
    container_dims: tuple
    container_contents: list = field(init=False)  # list of (TItem, top_left_square_location)
    container_grid: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.image is None:
            self.image = np.ones((64, 64, 3), np.uint8)
        self.container_grid = np.zeros(self.container_dims)
        self.container_contents = []

    def insert_item(self, item: TItem, location: tuple):  # insert an item at a location into the container
        self.container_contents.append((item, location))
        i, j = location

        height, width = item.dim
        self.container_grid[i:i+height, j:j+width] = 1

        # hopefully the bound checking will be done before we get here
        # and presumably I wont have to check if a spot is already filled, as if it is it is the fault of the
        # image reader.

        return

    def list_contents(self, children=False) -> list:  # list of all items and their locations in its relative container

        if not self.container_contents:  # return None if empty
            return None

        if not children:  # container's children will not be listed as well
            return self.container_contents

        else:
            list_of_contents = self.container_contents.copy()
            for item, location in self.container_contents:
                if type(item) == TContainerItem:
                    contents = item.list_contents(children=True)
                    list_of_contents += contents

            return list_of_contents

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
    unallocated_containers = []

    def expand_stash(self,stash_container_candidate):
        # used for when a stash screenshot is converted to a temporary TContainerItem.
        # Must:
        # 1) Attempt to determine where the screenshot fits in the stash.
        # 2) Ensure no lines are double counted between separate screenshots
        # 3) Change own dimensions
        # 4) Resize own grid
        # 5) Copy candidate grid into the new space
        pass

    def change_dims(self, new_dims):
        self.dim = new_dims
        self._resize_grid(new_dims)

    def _resize_grid(self, new_dims: tuple, bottom=True):
        pass

    def allocate_containers(self):

        """
        Due to the ability to input container internals BEFORE the container itself is inputted, there is a need for
        an "unallocated_containers" list to hold these internal containers, until they can (attempt to) be assigned
        to an appropriate container of the correct type.
        There could also be cases (such as heavy backpack nesting etc) where the user wont bother to input all nested
        layers, in this case it is useful to have a space so the internals can still be listed without requiring the
        external container to be added.
        """

        if not self.unallocated_containers:
            print("Allocate Containers: No unallocated Containers")
            return

        # create list of all empty_containers in the stashes contents (check inside of children as well)
        all_items = self.list_contents(children=True)
        empty_containers = []
        for item, location in all_items:
            if type(item) == TContainerItem:
                contents = item.list_contents(children=False)  # determine if contents are empty
                if not contents:
                    empty_containers.append(item)

        number_unallocated = len(self.unallocated_containers)
        print(f"Allocate Containers: {number_unallocated} unallocated containers")

        if not empty_containers:
            print("Allocate Containers: No empty containers detected")
            return

        for unallocated_container in self.unallocated_containers:
            for empty_container in empty_containers.copy():

                if empty_container.container_dims == unallocated_container.container_dims:
                    empty_container.container_contents = unallocated_container.container_contents
                    empty_container.container_grid = unallocated_container.container_grid

                    empty_containers.remove(empty_container)
                    print(f"Allocate Containers: {unallocated_container.name} --> {empty_container.name}")
                    break

        number_unallocated = len(self.unallocated_containers)
        print(f"Allocate Containers: {number_unallocated} unallocated containers remaining")

        return

    def list_contents(self, children=False, unallocated=True) -> Union[list, None]:

        # The TStash version has option for listing unallocated containers
        if unallocated:
            unallocated = self.unallocated_containers.copy()
        else:
            unallocated = []
        all_stash_contents = self.container_contents.copy() + unallocated

        if not all_stash_contents:  # return None if empty
            return None

        if not children:  # container's children will not be listed as well
            return all_stash_contents

        else:
            list_of_contents = all_stash_contents
            for item, location in all_stash_contents: # recursively gets contains from any containers nested.
                if type(item) == TContainerItem:
                    contents = item.list_contents(children=True)
                    list_of_contents += contents

            return list_of_contents

    def add_unallocated_container(self, input_container: "TContainerItem"):
        self.unallocated_containers.append((input_container, 9999))


if __name__ == "__main__":  # debug purposes
    test_item = TItem("ItemName", 1, (1, 1), False)
    print(test_item)
    test_container = TContainerItem("ContainerName", 1, (1, 1), False, (5, 5))
    print(test_container)
    stash = TStash()
    print(stash)
    print()


