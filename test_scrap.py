import requests
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import display


if __name__ == '__main__':
    r_total = requests.get('https://en.wikipedia.org/wiki/Central_limit_theorem#Multidimensional_CLT')
    if r_total.ok:
        print(r_total)
        # print(r_total.text)

        soup = BeautifulSoup(r_total.content, 'html.parser')
        print("\n\n\nSOUP:")
        # print(soup)
        # price = soup.find("img")
        # print(price)

        for img in soup.findAll('p'):
            print(img)
    else:
        print("Failed")
        print(r_total)
        print(r_total.reason)
        df_total = None
