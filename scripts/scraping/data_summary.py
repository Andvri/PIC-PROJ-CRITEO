from os import path
import pandas as pd
from IPython.display import display


if __name__ == "__main__":
    working_directory = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))

    df = pd.read_csv(working_directory + "/sources/data_train_with_parent.csv", header=0, index_col=0)

    print("\nDataframe:")
    display(df.head())

    print("\nNon-empty samples:")
    display(df[~df.description.isna()])

    print("\nInfo:")
    display(df.info())

    for cat in df.category.unique():
        print(f"\nCategory: {cat}\n"
              f"{df[~df.description.isna() & (df.category == cat)].shape[0]} "
              f"non-empty samples out of {df[df.category == cat].shape[0]}")
