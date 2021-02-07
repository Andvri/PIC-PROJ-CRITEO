#!/usr/bin/python3
import sys, getopt
import os.path
from os import path

def generate_categories(data, level):
    categories = []
    # Remove first line
    data.pop(0)
    for line in data:
        categories_line = line.split('>')
        i = 0
        for category in categories_line:
            if i < level and not category.strip()  in categories:
                categories.append(category.strip())
            i+=1
    return categories

def generate_categories_with_parent(data, level):
    categories = []
    # Remove first line
    data.pop(0)
    for line in data:
        categories_line = ' > '.join([category.strip() for category in  line.split('>')[:level]])
        
        if  not categories_line  in categories:
            categories.append(categories_line)
    return categories

def write_categories_file(path, categories):
    print('Writing categories in: ' + path)
    with open(path, 'w') as f:
        for category in categories:
            f.write("%s\n" % category)
    print('Categories correctly written in: ' + path)

def main(argv):
    working_directory = path.dirname(path.dirname(path.realpath(__file__)))
    taxonomy_file_path = working_directory + '/sources/taxonomy.en-US.txt'
    level = 2
    try:
        opts, args = getopt.getopt(argv,"l:h",["level="])
    except getopt.GetoptError:
        print(f'Depth level {level} is taken as default')
    for opt, arg in opts:
        if opt == '-h':
            print('build-categories.py -l <level>')
            sys.exit()
        elif opt in ("-l", "--level"):
            level = int(arg)

    print ("Depth level: " + str(level))
    if path.exists(taxonomy_file_path):
        print("File Exists:" + taxonomy_file_path)
    else:
        print(f'File ({taxonomy_file_path}) not exists')
        sys.exit()


    
    taxonomy_data = open(taxonomy_file_path, "r")
    data = taxonomy_data.readlines()
    categories = generate_categories(data, level)
    categories_file_path = working_directory + '/sources/categories/categories-level' + str(level) + '.txt'
    write_categories_file(categories_file_path, categories)

    categories = generate_categories_with_parent(data, level)
    categories_file_path = working_directory + '/sources/categories/categories-with-parent-level' + str(level) + '.txt'
    write_categories_file(categories_file_path, categories)
    

if __name__ == "__main__":
    main(sys.argv[1:])
