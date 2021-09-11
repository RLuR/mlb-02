########################################################################################################################
# EX1
########################################################################################################################


'''
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...

(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters

(2) find duplicates and exclude them
    - remove only the unnecessary rows

(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutff value for idnetifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers

(4) impute missing values using the KNNImputer
    - including the excluded outliers!
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!

(5) find the class distribution
    - use the groupby() method

(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
'''


########################################################################################################################
# Solution
########################################################################################################################


# set pandas options to make sure you see all info when printing dfs
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.datasets import load_wine
from scipy.stats import norm
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():
    # remove unnecessary header and footer file
    data = pd.read_csv("data/wine_exercise.csv", skiprows=1, skipfooter=1, sep=";")
    #### Repair dataset ####
    data["ash"] = data["ash"].replace("missing", np.nan)
    data["malic_acid"] = data["malic_acid"].replace("-999", np.nan)
    data["proline"] = data["proline"].replace(0.0, np.nan)
    # for now drop problematic rows
    badline = data.iloc[50]
    badline2 = data.iloc[142]
    badline3 = data.iloc[166]
    data = data.drop([50, 142, 166], axis=0)
    # fix commas
    data = data.astype(str)
    data = data.apply(lambda x: x.str.replace(',', '.'))
    # split country and age
    data[['country', 'age']] = data['country-age'].str.split("-", expand=True)
    data['age'] = data['age'].str.replace('years', '')
    data = data.drop("country-age", axis=1)
    # saving and reading get numeric Dtypes for all now numeric columns
    data.to_csv("temp.csv", index=False, sep=";")
    data = pd.read_csv("temp.csv", sep=";")

    # find and fix typos
    print("Unique Seasons:")
    print(data["season"].unique())
    data["season"] = data["season"].replace("spring", "SPRING")
    data["season"] = data["season"].replace("aut", "AUTUMN")
    print("Unique Seasons after data fixing:")
    print(data["season"].unique())

    print("Unique Countries:")
    print(data["country"].unique())


    # Remove duplicates
    print("Duplicate Rows:\n")
    duplicate_rows = data.duplicated()
    print(duplicate_rows[duplicate_rows == True])

    data = data.drop_duplicates()

    # Find and remove outliers
    data["magnesium"] = remove_outliers(data["magnesium"])
    data["total_phenols"] = remove_outliers(data["total_phenols"])

    # Impute data

    original_headers = load_wine()["feature_names"]

    print(f"Amount of missing data before imputation: {data.isnull().sum().sum()}")

    imputer = KNNImputer(n_neighbors=5)
    data[original_headers] = imputer.fit_transform(data[original_headers])

    print(f"Amount of missing data after imputation: {data.isnull().sum().sum()}")

    # Class distributions

    print(data.groupby(['season','country'])['target'].value_counts())

    print(data.groupby('color')['magnesium'].describe())


def remove_outliers(column):
    mean, std = norm.fit(column)
    low_cutoff = norm.ppf(.0125, loc=mean, scale=std)
    high_cutoff = norm.ppf(.99875, loc=mean, scale=std)
    return column.apply(lambda x: x if low_cutoff <= x <= high_cutoff else np.nan)

if __name__ == "__main__":
    main()