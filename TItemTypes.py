"""
Base Class for all inventory items.
Most items will probably just be of this type, however a few potential sub class extensions could be useful, such as
weapons, bags, and cases.

"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import copy


# On start up, TDataCatalog will create an instance of every class.
# When an item is detected and created, in __init__ it will be compared to the catalog in name, image, dimensions
# If a match is found the candidate will have its attributes set to the values of the catalog's.

@dataclass
class TItem:

    name: str
    image: np.ndarray
    dim: tuple  # (rows,columns)
    rotated: bool

    def compare_to(self, candidate: "TItem") -> "TItem":
        """
        Compares a candidate TItem from the catalog to this item.
        Intended to be used on an item from an image,with candidate being a catalog item.
        Determine if the candidate has sufficient matching information to the catalog version.
        If so, it will return a copy of the cataloged version (which contains more information)
        with potential modifications. IE: rotated = True
        Returns None if no match
        """
        threshold = .80  # this is just an arbitrary value, can be changed as more testing is done.
        if self._compute_correlation(candidate) > threshold:
            return self._copy_contents(candidate)

        if self._compare_name():
            # return self._copy_contents(candidate)
            pass

        return None  # if no match is found will just return None

    def _compute_correlation(self, candidate: "TItem") -> float:
        """
        Computes correlation between images of self and candidate
        """

        self_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        candidate_hsv = cv2.cvtColor(candidate.image, cv2.COLOR_BGR2HSV)

        # settings
        h_bins = 50
        s_bins = 60
        hist_size = [h_bins, s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges  # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]

        # convert to histogram, and normalize so hists of both images are the same size.
        hist_self = cv2.calcHist([self_hsv], channels, None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist_self, hist_self, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_candidate = cv2.calcHist([candidate_hsv], channels, None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist_candidate, hist_candidate, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        correlation = cv2.compareHist(hist_self, hist_candidate, cv2.HISTCMP_CORREL)  # compute correlation

        return correlation

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
        self.container_grid = np.zeros(self.container_dims)
        self.container_contents = []

    def insert_item(self, item: TItem, location: tuple):  # insert an item at a location into the container
        self.container_contents.append(item)
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

    def add_unallocated_container(self, input_container: "TContainerItem"):
        self.unallocated_containers.append(input_container)


if __name__ == "__main__":  # debug purposes
    test_item = TItem("ItemName", 1, (1, 1), False)
    print(test_item)
    test_container = TContainerItem("ContainerName", 1, (1, 1), False, (5, 5))
    print(test_container)
    stash = TStash()
    print(stash)
    print()


