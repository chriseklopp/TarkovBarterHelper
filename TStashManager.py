"""
Singleton
Aptly named after the in-game storage, this class will contain, store, and manage all TItems detected and created.
Holds all inputted information. IE. if a function want to know what is in your inventory they will ask this guy.


Will also call for reading of images by TImageReader and receives its output.
Able to resolve conflicts between images and piece together an accurate recreation of a Stash.
"""


class TStashManager:
    def __init__(self):
        pass