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

# TODO: Make Refactoring


options = Options()
options.headless = True
options.add_argument("--window-size=50,50")

working_directory = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))

# Chromedriver must be in the root of the project
# https://chromedriver.chromium.org/downloads
DRIVER_PATH = working_directory + '/chromedriver'
urls_scraping = working_directory + '/sources/input-scrapping.json'


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
    for [url, category] in data:

        print('\nRetrieving: '+ url)

        show_browser = not bool([site for site in sites_show_browser if(site in url)])
        wait = 0
        if show_browser:
            options.headless = True
            wait = 2
        else:
            options.headless = False
            

        driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)
        driver.implicitly_wait(wait)
        driver.get(url)
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
            end_data.append('\n'.join([url, description, category]))
            print('Done')
        else:
            print('Description not found')

    #Write file
    with open(working_directory + '/sources/data.txt', 'w') as f:
        for d in end_data:
            f.write("%s\n\n" % d)


