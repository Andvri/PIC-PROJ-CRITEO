import re


def clean_descriptions(df):
    clean_space = re.compile(
        r"(<.*?>)|(\\t)|( +)|=|\^|\$|\*|\?|{|}|\[|\]|\|\\|\||\(|\)|/|_"
        r"|-|:|`|'|â€™|<|>|\"|(&gt)|(&lt)|%|\+|[0-9]+|✅"
    )
    clean_nothing = re.compile("[.;:!\'?,\"()\[\]]")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.description = df.description.apply(lambda x: re.sub(clean_nothing, '', x))
    df.description = df.description.apply(lambda x: re.sub(clean_space, ' ', x))

    return df
