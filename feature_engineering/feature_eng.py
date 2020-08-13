import pandas as pd
import numpy as np


# replace missing values
def vars_with_nan(df: pd.DataFrame) -> list:
    # make a list of the categorical variables that contain missing values
    vars_with_na = [var for var in df.columns if df[var].isnull().sum() > 1 and df[var].dtypes == 'O']
    return vars_with_na


# function to replace NA in categorical variables
def fill_categorical_na(df: pd.DataFrame, vars_with_na: list) -> pd.DataFrame:
    X = df.copy()
    X[vars_with_na] = df[vars_with_na].fillna('Missing')
    return X


def feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    num_vars = [var for var in df.columns if df[var].dtypes != 'O']
    x = df.copy()
    for var in num_vars:
        x[var] /= df[var].max()

    return x


def replace_rare_labels(df: pd.DataFrame, cat_vars: list, target: str)-> pd.DataFrame:
    def find_frequent_labels(d: pd.DataFrame, var: str, rare_perc: float):
        # finds the labels that are shared by more than a certain % of the houses in the dataset
        d = d.copy()
        tmp = d.groupby(var)[target].count() / len(d)
        return tmp[tmp > rare_perc].index

    df = df.copy()
    for var in cat_vars:
        frequent_ls = find_frequent_labels(df, var, 0.01)
        df[var] = np.where(df[var].isin(frequent_ls), df[var], 'Rare')

    return df


# this function will assign discrete values to the strings of the variables,
# so that the smaller value corresponds to the smaller mean of target
def replace_categoricals(df:pd.DataFrame, cat_vars:list)-> pd.DataFrame:
    def replace_categories(d: pd.DataFrame, var: str, target: str)-> pd.DataFrame:
        ordered_labels = d.groupby([var])[target].mean().sort_values().index
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
        d[var] = d[var].map(ordinal_label)

        return d

    for var in cat_vars:
        df = replace_categories(df, var, 'price')

    return df
