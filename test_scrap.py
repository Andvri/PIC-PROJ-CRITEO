import requests
from bs4 import BeautifulSoup
import time

# Url of the page we want to scrape:
urls = [
    "https://www.fnac.com/Pack-PC-Hybride-Microsoft-Surface-Go-2-10-5-Tactile-Intel-Pentium-Gold-8Go-RAM-128Go-SSD-Platine-Clavier-Type-Cover-Microsoft-Surface-Noir-pour-Microsoft-Surface-Go-2-Souris-Microsoft-Surface-Mobile-Platine/a14855208/w-4",
    "https://www.amazon.fr/gp/product/B07YW3Z8HH?pf_rd_r=JCFRPP39183M66ZYKZZF&pf_rd_p=ed1ef413-005c-474d-837a-434c7d76d0d9&pd_rd_r=66d752f3-3ab7-4435-b1f6-4b71e74f4b24&pd_rd_w=V1Yf4&pd_rd_wg=eAwpd&ref_=pd_gw_unk",
    "https://www.darty.com/nav/achat/petit_electromenager/expresso_cafetiere/expresso_avec_broyeur/krups_yy4046fd.html#dartyclic=soldes_et_bons_plans_3_4585321",
    "https://fr.shopping.rakuten.com/boutique/setdiscount"
]

if __name__ == "__main__":
    for url in urls:
        request = requests.get(url)
        
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
