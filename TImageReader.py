"""
Processes the screenshot(s) into detected TItems.
Contains logic for processing a singular screenshot into an ARRAY of TItems (this MAY be some custom type object)
(This array structure will probably be important for out of game accurate inventory representation, as well
as (most importantly) verification of detection accuracy, avoidance of double counting, and allocation of items to the
correct container.

Probably might not even have to be a class.
"""


class TImageReader:
    def __init__(self):
        pass
