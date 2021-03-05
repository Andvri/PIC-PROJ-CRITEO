from os import path
import time
import pandas as pd
import requests
from googlesearch import search


working_directory = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))

TIME_DELAY = 120
RESULTS_PER_CATEGORY = 1000
CATEGORIES = {
    "Electronics > Computers": {
        "request_list": [
            "Lenovo", "Apple",
            "HP", "Dell", "Acer", 
            "Asus", "Microsoft", "Alienware", "Legion"
        ],
        "general_search": "Computers"
    },
    "Electronics > Communications": {
        "request_list": [
            "Iphone", "Google Pixel",
            "Xiaomi", "Huawei", "Samsung", "OnePlus",
            "LG", "Nokia", "Motorola", "HTC"
        ],
        "general_search": "Smartphones"
    },
    "Apparel & Accessories > Jewelry": {
        "request_list": [
            "Necklace", "Bracelet",
            "Earrings", "Rings", "Amethyst",
            "Aquamarine", "Diamond", "Ruby",
            "Sapphire", "Golden", "Silver"
        ],
        "general_search": "Jewels"
    },
    "Vehicles & Parts > Vehicle Parts & Accessories": {
        "request_list": [
            "Bumper", "Radiator", "Spoiler",
            "Tires", "Rims", "Breaks",
            "Windshields", "Windows", "Headlights"
        ],
        "general_search": "Car parts"
    },
    "Software > Computer Software": {
        "request_list": [
            "PC games", "Console games", "Antivirus",
            "System", "Device drivers", "Multimedia",
            "Graphics", "Database", "Word Processors"
        ],
        "general_search": "Software"
    },
    "Apparel & Accessories > Clothing": {
        "request_list": [
            "Polo", "Dresses", "Hoodies",
            "T-shirts", "Flip-flops", "Shorts",
            "Skirts", "Jeans", "Shoes",
            "Coats", "Shirts", "Socks",
            "Suits", "Caps", "Scarfs",
            "Underwear", "Sweaters", "Jackets", "Hats"
        ],
        "general_search": "Clothes"
    },
    "Furniture > Chairs": {
        "request_list": [
            "Desk", "Rocking", "Office", "Garden", "Stool", "Armchair", "Baby", "Lounge", "Side"
        ],
        "general_search": "Chairs"
    },
    "Media > Books": {
        "request_list": [
            "Pride and Prejudice", "To Kill a Mockingbird", "Nineteen Eighty-Four", "Buddenbrooks", "The Grapes of Wrath", "Beloved", "The Code of the Woosters", "Dracula",
            "The Great Gatsby", "One Hundred Years of Solitude", "In Cold Blood", "Wide Sargasso Sea", "Brave New World", "I Capture The Castle", "Jane Eyre", "The Lord of the Rings",
            "Crime and Punishment", "The Secret History", "The Call of the Wild", "The Chrysalids", "Persuasion", "Moby-Dick", "The Lion, the Witch and the Wardrobe", "Catch-22",
            "To the Lighthouse", "The Death of the Heart", "Tess of the d'Urbervilles", "Frankenstein", "The Master and Margarita", "The Go-Between", "One Flew Over the Cuckoo's Nest",
            "The Adventures of Huckleberry Finn", "Great Expectations", "The Age of Innocence", "Things Fall Apart", "Middlemarch", "Midnight's Children", "The Iliad", "Vanity Fair",
            "Brideshead Revisited", "The Catcher in the Rye", "Alice’s Adventures in Wonderland", "The Mill on the Floss", "Barchester Towers", "Another Country", "Les Miserables",
            "Charlie and the Chocolate Factory", "The Outsiders", "The Count of Monte Cristo", "Ulysses", "East of Eden", "The Brothers Karamazov", "Lolita", "The Secret Garden",
            "Scoop", "A Tale of Two Cities", "Diary of a Nobody", "Anna Karenina", "The Betrothed", "Orlando", "Atlas Shrugged", "The Time Machine", "The Art of War", "The Forsyte Saga",
            "Travels with Charley", "Tropic of Cancer", "Women in Love", "Staying On", "The Wind in the Willows", "My Ántonia", "Wuthering Heights", "Perfume", "War and Peace", "Of Human Bondage",
            "Bleak House", "Lost Illusions", "Breakfast of Champions", "A Christmas Carol", "Silas Marner", "Mrs Dalloway", "Little Women", "The Sea, The Sea", "The Godfather", "The Castle", "I, Claudius",
            "Peter Pan", "A Confederacy of Dunces", "The Razor's Edge", "Lark Rise to Candleford", "The Return of the Native", "A Portrait of the Artist as a Young Man", "Heart of Darkness",
            "North and South", "The Handmaid's Tale", "Suite Francaise", "One Day in the Life of Ivan Denisovich", "What A Carve Up!", "Zen and the Art of Motorcycle Maintenance", "White Nights", "Hard Times",
            "THE SOUND AND THE FURY", "DARKNESS AT NOON", "SONS AND LOVERS", "UNDER THE VOLCANO", "THE WAY OF ALL FLESH", "I, CLAUDIUS", "AN AMERICAN TRAGEDY", "THE HEART IS A LONELY HUNTER", "SLAUGHTERHOUSE-FIVE",
            "INVISIBLE MAN", "NATIVE SON", "HENDERSON THE RAIN KING", "APPOINTMENT IN SAMARRA", "U.S.A. ", "WINESBURG, OHIO", "A PASSAGE TO INDIA", "THE WINGS OF THE DOVE", "THE AMBASSADORS", "TENDER IS THE NIGHT",
            "THE STUDS LONIGAN TRILOGY", "THE GOOD SOLDIER", "ANIMAL FARM", "THE GOLDEN BOWL", "SISTER CARRIE", "A HANDFUL OF DUST", "AS I LAY DYING", "ALL THE KING’S MEN", "THE BRIDGE OF SAN LUIS REY", "HOWARDS END",
            "GO TELL IT ON THE MOUNTAIN", "THE HEART OF THE MATTER", "LORD OF THE FLIES", "DELIVERANCE", "A DANCE TO THE MUSIC OF TIME", "POINT COUNTER POINT", "THE SUN ALSO RISES", "THE SECRET AGENT", "NOSTROMO",
            "THE RAINBOW", "THE NAKED AND THE DEAD", "PORTNOY’S COMPLAINT", "PALE FIRE", "LIGHT IN AUGUST", "ON THE ROAD", "THE MALTESE FALCON", "PARADE’S END", "ZULEIKA DOBSON", "THE MOVIEGOER", "MAIN STREET",
            "DEATH COMES FOR THE ARCHBISHOP", "FROM HERE TO ETERNITY", "THE WAPSHOT CHRONICLES", "A CLOCKWORK ORANGE", "THE HOUSE OF MIRTH", "THE ALEXANDRIA QUARTET", "A HIGH WIND IN JAMAICA", "A HOUSE FOR MR BISWAS",
            "THE DAY OF THE LOCUST", "A FAREWELL TO ARMS", "THE PRIME OF MISS JEAN BRODIE", "FINNEGANS WAKE", "KIM", "A ROOM WITH A VIEW", "THE ADVENTURES OF AUGIE MARCH", "ANGLE OF REPOSE", "A BEND IN THE RIVER",
            "LORD JIM", "RAGTIME", "THE OLD WIVES’ TALE", "LOVING", "MIDNIGHT’S CHILDREN", "TOBACCO ROAD", "IRONWEED", "THE MAGUS", "UNDER THE NET", "SOPHIE’S CHOICE", "THE SHELTERING SKY", "THE POSTMAN ALWAYS RINGS TWICE",
            "THE GINGER MAN", "THE MAGNIFICENT AMBERSONS", "Harry Potter", "Tom Sawyer", "Arsen Lupin", "Sherlock Holmes", "Hercule Poirot",
            "Agatha Christie", "Georges Simenon", "Leo Tolstoy", "Stephen King", "Paulo Coelho", "Edgar Wallace", "J. R. R. Tolkien"
        ],
        "general_search": "Books"
    },
    "Home & Garden > Plants": {
        "request_list": [
            "Magnolia", "Moss", "Green Algae", "Peace Lily", "Fern", "Ginkgo", "Daffodil", "Spider Plant", "English Ivy", "Conifer", "Shrub",
            "Spruce", "Rubber Plant", "Sago Palm", "White Lily", "Golden Pothos", "Mass Cane", "Aloe", "Succulent", "Flowers", "Cactus", "Fir"
        ],
        "general_search": "Plants"
    },
    "Luggage & Bags > Suitcases": {
        "request_list": [
            "Hardside", "Softside", "Leather", "Wheeled", "Hand", "Aluminum", "Polycarbonate", "Polypropylene", "ABS"
        ],
        "general_search": "Suitcases"
    },
    "Cameras & Optics > Cameras": {
        "request_list": [
            "Compact", "DSLR", "Mirrorless", "Action", "360", "Medium Format", "Traditional Film", "GoPro",
            "Canon", "Nikon", "Panasonic", "Sony", "Fujifilm"
        ],
        "general_search": "Cameras"
    }
}


if __name__ == "__main__":
    urls = []
    categories = []

    for category, search_req in CATEGORIES.items():
        print(f"\nExtracting {category.split(' > ')[-1]} URLs...")
        # Compute nuber of URLs per category result:
        nb_results = RESULTS_PER_CATEGORY // (len(search_req["request_list"]) + 1) + 1
        print(f"{nb_results} results per request")
        
        # Make a general category request:
        try:
            search_results = list(search(search_req["general_search"] + " buy online", num_results=nb_results, lang="en"))[:nb_results]
        except requests.exceptions.HTTPError:
            print(f"Stopped at {category.split(' > ')[-1]} category")
            # Save results:
            df = pd.DataFrame(data={"URL": urls, "category": categories})
            df.to_csv(working_directory + "/sources/input_scraping_train_with_parent.csv", mode='a', header=False)
            exit()
        
        urls.extend(search_results)
        categories.extend([category] * len(search_results))

        time.sleep(TIME_DELAY)

        # Make some specific requests:
        for req in search_req["request_list"]:
            print(f"Extracting {req} URLs...")
            search_results = list(search(req + " " + search_req["general_search"] + " buy online", num_results=nb_results, lang="en"))[:nb_results]
            urls.extend(search_results)
            categories.extend([category] * len(search_results))

            time.sleep(TIME_DELAY)

    # Save results:
    df = pd.DataFrame(data={"URL": urls, "category": categories})
    df.to_csv(working_directory + "/sources/input_scraping_train_with_parent.csv", mode='a', header=False)
