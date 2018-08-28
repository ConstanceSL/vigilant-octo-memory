import pandas as pd
from os import walk
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
IMPORTING THE DATA
"""


def sentence_to_list(column):
    return column.split(' ')


def compile_data(dir_path):
    """
    Compiles the data from the folder where all the processed data is stored
    :param paths: either a list of paths or a single path
    :return: a data frame with the data from the csv files combined together
    """
    files = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        for file in filenames:
            if file[0]!='.':
                files.append(join(dir_path, file))
    data = []
    for path in files:
        try:
            df = pd.read_csv(path, engine='c')
        except Exception:
            df = pd.read_csv(path, engine='python')
        df.fillna(value=np.nan, inplace=True)
        df.dropna(subset=['text', 'id_number', 'text_lemmatized']).reset_index(drop=True)
        df['text_lemmatized'] = df['text_lemmatized'].astype(str).apply(sentence_to_list)
        data.append(df)
    return pd.concat(data).reset_index(drop=True)