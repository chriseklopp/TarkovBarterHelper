# TarkovBarterHelper
Escape from Tarkov Inventory Analyzer and Barter Helper

** This project has been rewritten in C++ at https://github.com/chriseklopp/TarkovInventoryAssistant.
** All further work on this project will happen there instead. Leaving this old repo up for archival purposes.



Motivation:
------------------------------------------------------------------------------------------------------------------------
Escape from Tarkov doesn't have a method of itemizing your hideout Stash (item storage). The only useful information the
game gives you is your total stash value to the Traders. With the complex barter system in game, it can be a pain to
keep track of what items you have, need, or should sell to the Traders or on the flea market. This tool to itemize an
inventory into a readable list (and potentially other functions) will make inventory management a lot easier and less of
a headache.
------------------------------------------------------------------------------------------------------------------------


Outline:
------------------------------------------------------------------------------------------------------------------------
Use template matching to identify and catalog items from screen shots of a Tarkov stash.
Should be able to recognize all items.
Should be able to avoid  double counting rows if they are present in multiple screenshots
Able to calculate available barters through traders, as well as highlight barters that are a net $ value gain.

VERSION 1:
Correctly Itemize all items in a single screenshot.

VERSION 2:
Correctly Itemize all items in MULTIPLE screenshots.

VERSION 3:
QT interface. Select screenshots and visually show details about inventory analysis.

VERSION 4:
Calculate and show available barters and recommend optimal traders selection for selling.

VERSION 5:
(This would be beyond the expected scope of the project, due to the potential issues with getting this information)
Get and use Flea Market Data to determine items to post on market VS selling to vendors. Detect cases where an item
could be bought off the flea market and used in a barter that would cause a net gain in value.
