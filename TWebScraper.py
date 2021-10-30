"""
This probably wont need to run at all during normal use of the program, but will have to be run periodically to update
the local catalog of items.


Will scrape https://escapefromtarkov.fandom.com/wiki/Loot for item information.
Eventually will probably scrape flea market data from another website.
"""


import bs4
import pandas as pd
import requests

# TODO: This class is getting a bit big, split it if it becomes too unwieldy.


class TWebScraper:
    def __init__(self, item=True, price=True):
        if item and price:
            # self.supplemental_item_information_gatherer("https://escapefromtarkov.fandom.com/wiki/Folder_with_intelligence")
            self.catalog_builder("https://escapefromtarkov.fandom.com/wiki/Loot")
            # self.catalog_builder("https://escapefromtarkov.fandom.com/wiki/Containers")
            self.price_catalog_builder()
        elif item:
            self.catalog_builder()
        elif price:
            self.price_catalog_builder()

    def catalog_builder(self, url: str):
        """
         https://escapefromtarkov.fandom.com/wiki/Loot
        """

        # TODO: Make this generate an appropriate file structure to save each table and its corresponding images.
        # TODO: This will allow for easier searching and reading back in to the main program.
        source = requests.get(url).text
        soup = bs4.BeautifulSoup(source, 'lxml')
        page_tables = soup.findAll('table', class_='wikitable sortable')
        for table in page_tables:
            self.table_builder(table)

    def table_builder(self, table: bs4.element.Tag):
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

        col_names = master_list.pop(0)  # first entry is the table header.
        extracted_names = []
        for item in col_names:
            extracted_names.append(item['text'][0])

        table_df = pd.DataFrame(master_list, columns=extracted_names)
        table_df = self.table_cleaner(table_df)

        if "dims" not in table_df.columns:
            if "Item_url" in table_df.columns:
                table_df["dims"] = table_df['Item_url'].apply(self.supplemental_item_information_gatherer)
            else:
                print("No Item_url column found")

        # TODO: Have function here to download the images from each Image_url.
        print("woww")

    @staticmethod
    def image_downloader(df: pd.DataFrame):
        pass

    def price_catalog_builder(self):
        pass

    @staticmethod
    def supplemental_item_information_gatherer(url: str) -> dict:
        # This name is terrible.
        # Some information needs to be grabbed from an items specific page, (specifically we want its dimensions)
        # This will act on an Item_url and return the necessary information as a dict.
        print(f"Dimensions being gathered for:{url}")
        information_dict = {}
        source = requests.get(url).text
        soup = bs4.BeautifulSoup(source, 'lxml')
        item_table = soup.find('table', class_='va-infobox')
        table_rows = item_table.findAll('tr')
        for element in table_rows:
            infobox_labels = element.find('td',class_='va-infobox-label')  # type: bs4.element.Tag
            if infobox_labels:
                if "Grid" in infobox_labels.string:
                    information_dict['dims'] = element.find('td', class_='va-infobox-content').string
                    break
                else:
                    continue

        return information_dict

    def table_cleaner(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Potentially implement a better solution to handling 'Other' column types
        # While different tables on the wiki may have different information, and ordering, there is some specific
        # information types we are looking from each entry that is shared among each table.
        # Specifically we want: name, image url, item url, and any other identifiable information on the item.
        # 1) classify columns
        # 2) Extract useful information from them
        # 3) return new cleaner df.

        cleaned_df = pd.DataFrame()
        column_names = df.columns
        classifications = self.column_classifier(df)
        for col_number, col_type in enumerate(classifications):
            specified_column = df.iloc[:, col_number]

            if col_type == "Name":
                cleaned_df["Name"] = specified_column.apply(self.information_extractor, tag="title")
                cleaned_df["Item_url"] = specified_column.apply(self.information_extractor, tag="href")
                cleaned_df["Item_url"] = cleaned_df["Item_url"].apply(self.url_fixer)

            elif col_type == "Image":
                cleaned_df["Image_url"] = specified_column.apply(self.information_extractor, tag="data-src",
                                                                 alt_tag="src")

            elif col_type == "Identifier":
                cleaned_df[column_names[col_number]] = specified_column.apply(self.information_extractor, tag="text")

            elif col_type == "Other":
                cleaned_df[column_names[col_number]] = specified_column.apply(self.information_extractor, tag="text")

            else:
                raise NameError('Incorrect column classifier')
        return cleaned_df

    @staticmethod
    def url_fixer(url: str) -> str:
        # Item paths only have a relative path on the wiki, so we re-add the rest of the url.
        if "https" not in url:
            return "https://escapefromtarkov.fandom.com" + url
        else:
            return url

    @staticmethod
    def column_classifier(df: pd.DataFrame) -> list:
        # identify each column as, Image, Name, Identifier, or OTHER (other is any non-useful field, ie. Notes)
        classifications = []
        for column_name in df.columns:
            col_sample = df[column_name][0]
            if "Name" in column_name:
                name_value = col_sample.get('title')

                if not name_value:
                    print("A classification error may have occurred")
                    classifications.append("Other")

                else:
                    classifications.append("Name")
                continue

            detected_data_src = col_sample.get("data-src")  # sometimes only one of these is present.  Its infuriating.
            detected_src = col_sample.get("src")
            if detected_data_src or detected_src:
                classifications.append("Image")
                continue

            detected_text = col_sample.get("text")
            if detected_text:
                if len(detected_text) < 3:
                    classifications.append("Identifier")
                else:
                    classifications.append("Other")
            else:
                classifications.append("Other")
            continue

        return classifications

    def information_extractor(self, info_dict: dict, tag: str, alt_tag=None) -> str:
        # extract information from a dictionary key from each row in a df column.
        # if alt tag provided, if tag returns None, will try the alt_tag instead.
        if alt_tag:
            value = self.information_extractor(info_dict, tag)
            if not value:
                value = self.information_extractor(info_dict, alt_tag)
            return value

        value = info_dict.get(tag)

        if not value:  # value is None if this key DNE
            print(f"{tag} tag not found")  # this will not happen if tag isn't found on first entry
            return None

        if type(value) == list:
            return value[0]

        return value

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
