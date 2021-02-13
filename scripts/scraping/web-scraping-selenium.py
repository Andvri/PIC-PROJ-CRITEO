from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os.path
from os import path
from bs4 import BeautifulSoup
import time
import os.path
from os import path
import json
import numpy as np
import pandas as pd
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
input_csv_path = working_directory + '/sources/input_scraping_en.csv'


sites_show_browser = [
    'amazon',
    'fnac',
    'darty',
    'rakuten'
]


# Url of the page we want to scrape:
with open(urls_scraping) as f:
    data = json.load(f)



if __name__ == "__main__":
    end_data = []

    # # Save data in .csv format:
    # df = pd.DataFrame(data=data, columns=["URL", "category"])
    # df.to_csv(working_directory + "/sources/input_scraping_with_parent.csv")
    
    # df.category = df.category.apply(lambda x: x.split(">")[-1].strip())
    # display(df)
    # df.to_csv(working_directory + "/sources/input_scraping.csv")
    
    # Load an input csv:
    df = pd.read_csv(input_csv_path, header=0, index_col=0)
    display(df)

    # The dataframe to store scraped data:
    df_data = df.copy()
    df_data.rename(columns={"URL": "description"}, inplace=True)

    for index, row in df.iterrows():
        print('\nRetrieving: '+ row.URL)

        show_browser = not bool([site for site in sites_show_browser if(site in row.URL)])
        wait = 0
        if show_browser:
            options.headless = True
            wait = 2
        else:
            options.headless = False
            

        driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
        driver.implicitly_wait(wait)
        driver.get(row.URL)
        # Now, we could simply apply bs4 to request variable
        soup = BeautifulSoup(driver.page_source, 'html.parser')


        driver.quit()


        time.sleep(1)
        
        """Scrape page description."""
        description = None
        if soup.find("meta", {"name": "description"}):
            description = soup.find("meta", {"name": "description"})['content']
        elif soup.find("meta", {"name": "og:description"}):
            description = soup.find("meta", {"name": "og:description"})['content']
        elif soup.find("meta", {"name": "twitter:description"}):
            description = soup.find("meta", {"name": "twitter:description"})['content']
        
        if description:
            end_data.append('\n'.join([row.URL, description, row.category]))
            df_data.loc[index, 'description'] = description # add descrption
            print('Done')
        else:
            df_data.loc[index, 'description'] = None # leave an empty case
            print('Description not found')

    # Save collected data in csv format:
    df_data.to_csv(working_directory + "/sources/data_en.csv")

    #Write file
    with open(working_directory + '/sources/data_en.txt', 'w') as f:
        for d in end_data:
            f.write("%s\n\n" % d)
