from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from os import path
import json
import numpy as np
import pandas as pd
from googlesearch import search
from IPython.display import display

# TODO: Make Refactoring


options = Options()
options.headless = True
options.add_argument("--window-size=50,50")

working_directory = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))

# Chromedriver must be in the root of the project
# https://chromedriver.chromium.org/downloads
DRIVER_PATH = working_directory + '/chromedriver'
urls_scraping = working_directory + '/sources/input-scrapping.json'
input_csv_path = working_directory + '/sources/input_scraping_train_with_parent.csv'
output_csv_path = working_directory + '/sources/data_train_with_parent.csv'

sites_show_browser = [
    'amazon',
    'fnac',
    'darty',
    'rakuten',
    'store.acer',
    'dell'
]

# When to save intermideate results:
CHECKPOINT = 20
PREVIOUS_CHECKPOINT = 0
# Time delay:
TIME_DELAY = 1

# Url of the page we want to scrape:
# with open(urls_scraping) as f:
#     data = json.load(f)


if __name__ == "__main__":
    end_data = []

    # Load an input csv:
    df = pd.read_csv(input_csv_path, header=0, index_col=0)
    df.drop_duplicates(inplace=True, ignore_index=True)
    display(df)

    # The dataframe to store scraped data:
    df_data = df.copy()
    df_data.rename(columns={"URL": "description"}, inplace=True)

    if PREVIOUS_CHECKPOINT:
        # Previous results:
        df_prev_output = pd.read_csv(output_csv_path, header=0, index_col=0)
        df_data[:PREVIOUS_CHECKPOINT] = df_prev_output[:PREVIOUS_CHECKPOINT]
        display(df_data)

    for index, row in df[PREVIOUS_CHECKPOINT:].iterrows():
        print(f'\n{index} Retrieving: '+ row.URL)

        show_browser = not bool([site for site in sites_show_browser if site in row.URL])
        wait = 0
        if show_browser:
            options.headless = True
            wait = 10
        else:
            options.headless = True
            wait = 3

        description = None
        try:
            driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
            driver.implicitly_wait(wait)
            driver.get(row.URL)
            # Now, we could simply apply bs4 to request variable
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            driver.quit()

            time.sleep(TIME_DELAY)
            
            """Scrape page description."""
            if soup.find("meta", {"name": "description"}):
                description = soup.find("meta", {"name": "description"}).get('content', None)
            elif soup.find("meta", {"name": "og:description"}):
                description = soup.find("meta", {"name": "og:description"}).get('content', None)
            elif soup.find("meta", {"name": "twitter:description"}):
                description = soup.find("meta", {"name": "twitter:description"}).get('content', None)
            elif soup.find("meta", {"property": "description"}):
                description = soup.find("meta", {"property": "description"}).get('content', None)
            elif soup.find("meta", {"property": "og:description"}):
                description = soup.find("meta", {"property": "og:description"}).get('content', None)
            elif soup.find("meta", {"property": "twitter:description"}):
                description = soup.find("meta", {"property": "twitter:description"}).get('content', None)
        except:
            print("Connection problem")

        if description:
            end_data.append('\n'.join([row.URL, description, row.category]))
            df_data.loc[index, 'description'] = description # add description
            print('Done')
        else:
            df_data.loc[index, 'description'] = None # leave an empty case
            print('Description not found')

        # Checkpoint save:
        if not (index + 1) % CHECKPOINT:
            print("Checkpoint save...")
            df_data.iloc[:index + 1, :].to_csv(working_directory + "/sources/data_train_with_parent.csv")

    # Save collected data in csv format:
    df_data.to_csv(working_directory + "/sources/data_train_with_parent.csv")

    # Write file:
    with open(working_directory + '/sources/data_train_with_parent.txt', 'w') as f:
        for d in end_data:
            f.write("%s\n\n" % d)
