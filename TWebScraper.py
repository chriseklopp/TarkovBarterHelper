"""
This probably wont need to run at all during normal use of the program, but will have to be run periodically to update
the local catalog of items.


Will scrape https://escapefromtarkov.fandom.com/wiki/Loot for item information.
Eventually will probably scrape flea market data from another website.
"""

import bs4
import numpy as np
import pandas as pd
import requests


class TWebScraper:
    def __init__(self, item=True, price=True):
        if item and price:
            self.item_catalog_builder()
            self.price_catalog_builder()
        elif item:
            self.item_catalog_builder()
        elif price:
            self.price_catalog_builder()

    def item_catalog_builder(self):

        """
         https://escapefromtarkov.fandom.com/wiki/Loot
        """

        source = requests.get("https://escapefromtarkov.fandom.com/wiki/Loot").text
        soup = bs4.BeautifulSoup(source, 'lxml')
        table = soup.find('table', class_='wikitable sortable')
        table_body = table.find("tbody")

        table_rows = table_body.findAll('tr')
        # Read table row by row, within each row read column wise.

        master_list = []
        for row in table_rows:
            row_contents = []
            columns = row.findAll(recursive=False)  # type: bs4.element.ResultSet
            for element in columns:  # each element is type bs4.element.Tag
                row_contents.append(self.element_parser(element))
            row_tuple = tuple(row_contents)
            master_list.append(row_tuple)

        col_names = master_list.pop(0)
        extracted_names = []
        for item in col_names:
            extracted_names.append(item['text'][0])

        table_df = pd.DataFrame(master_list, columns=extracted_names)
        print()

    def price_catalog_builder(self):
        pass

    @staticmethod
    def table_cleaner():
        # While different tables on the wiki may have different information, and ordering, there is some specific
        # information types we are looking from each entry that is shared among each table.
        # Specifically we want: name, image url, and any other identifiable information on the item.
        pass

    @staticmethod
    def image_element():
        pass

    @staticmethod
    def name_element():
        pass

    @staticmethod
    def identifier_element():
        pass

    @staticmethod
    def element_parser(element: bs4.element.Tag) -> dict:
        print("-------------------------")
        # Parse attributes and text from an element's descendants into a dictionary

        element_contents = {}
        for descendant in element.descendants:
            if descendant.name == 'a' or descendant.name == 'img':  # if it is a dictionary type container.
                for key, value in descendant.attrs.items():
                    element_contents[key] = value

            else:  # for not dictionary types, likely just text or empty garbage.

                text = descendant.string
                if not text:  # check for NoneType
                    continue
                text = text.strip("\n")

                if not text:
                    continue
                text = text.strip()

                if text:  # one last time ensure text still exists.
                    if "text" not in element_contents:
                        element_contents["text"] = []
                    element_contents["text"].append(text)

        # element_contents["text"] = set(element_contents["text"])

        print(element_contents)
        return element_contents


if __name__ == "__main__":  # run this to update the catalog from the wiki.
    TWebScraper()
