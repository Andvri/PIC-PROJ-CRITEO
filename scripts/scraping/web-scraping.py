#!/usr/bin/python3
import requests
from bs4 import BeautifulSoup
import time
import os.path
from os import path
import json


working_directory = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
urls_scraping = working_directory + '/sources/input-scrapping.json'
headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

# Url of the page we want to scrape:
with open(urls_scraping) as f:
    data = json.load(f)

if __name__ == "__main__":
    for [url, category] in data:
        print('Retrieving: '+ url)
        request = requests.get(url, headers)
        
        if request.ok:
            # this is just to ensure that the page is loaded:
            time.sleep(1)
            
            # Now, we could simply apply bs4 to request variable 
            soup = BeautifulSoup(request.content, "html.parser")
            """Scrape page description."""
            description = None
            if soup.find("meta", {"name": "description"}):
                description = soup.find("meta", {"name": "description"})['content']
            elif soup.find("meta", {"name": "og:description"}):
                description = soup.find("meta", {"name": "og:description"})['content']
            elif soup.find("meta", {"name": "twitter:description"}):
                description = soup.find("meta", {"name": "twitter:description"})['content']
            
            print(description)
