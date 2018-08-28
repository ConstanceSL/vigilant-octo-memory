import pandas as pd
from os import walk
from os.path import join
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
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
        dates = [date for date in list(set(df['date'])) if len(date) == 10]
        df = df[df['date'].isin(dates)]
        df.fillna(value=np.nan, inplace=True)
        df = df.dropna(subset=['text', 'id_number', 'text_lemmatized']).reset_index(drop=True)
        df['text_lemmatized'] = df['text_lemmatized'].astype(str).apply(sentence_to_list)
        data.append(df)
    return pd.concat(data).reset_index(drop=True)


"""
WORD EMBEDDINGS
"""


def train_model(df, model_name, iterations):
    """
    Creates and saves a word embedding model
    :param df: Data to fit the model on
    :param model_name: Name to give the model to save it
    :param iterations: Number of iterations to train the model
    :return: Trained model
    """
    sentences = df['text_lemmatized'].tolist()
    model = Word2Vec(sentences, min_count=10, size=300, window=5, workers=3, iter=iterations)
    model.wv.save_word2vec_format('./word_embeddings/models/' + model_name + '.bin')
    model.wv.save_word2vec_format('./word_embeddings/models/' + model_name + '.txt', binary=False)
    print('{} saved'.format(model_name))
    return model


def plot_closest_words(model, word, title, figure_path, colour):
    """
    Plots the 20 words closest to a given word
    :param model: Word embedding model
    :param word: Word to plot
    :param title: Title of the grah
    :param figure_path: Where to save the figure
    :param colour: colour of the plot
    :return: None, but saves the figure to file
    """
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word, topn=20)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    fig, axis = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    sns.regplot(x=x_coords, y=y_coords, fit_reg=False, marker="o", color=colour, scatter_kws={'s': 80}, ax=axis)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(7, -5), textcoords='offset points', size='medium')
    plt.xlim(x_coords.min() - 50, x_coords.max() + 50)
    plt.ylim(y_coords.min() - 50, y_coords.max() + 50)
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.savefig(figure_path, bbox_inches='tight')
    return None


def word_embeddings(df, list_of_words):
    # Training the models
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    print('Training the models')
    model_pro = train_model(df[df['orientation'] == 'pro'], 'pro_' + date, 100)
    model_against = train_model(df[df['orientation'] == 'against'], 'against_' + date, 100)
    # Plotting the words
    print('Plotting the list of words')
    for word in list_of_words:
        plot_closest_words(model_pro, word, 'Words similar to ' + word + ' (pro tweets)',
                           './word_embeddings/figures/' + word + '_pro_' + date + '.pdf', 'C3')
        plot_closest_words(model_against, word, 'Words similar to ' + word + ' (against tweets)',
                           './word_embeddings/figures/' + word + '_against_' + date + '.pdf', 'C0')
    print('Figures saved to file')




"""
GENERAL FUNCTION
"""


def import_tweets_and_do_embeddings():
    """
    General function that loads the data, fits the models, and plots words similarities
    Prompts the user to enter the path to the directory where the data is stored and of where to save the results
    :return: None
    """
    path = input('Please enter the path of the project directory: ')
    os.chdir(path)
    df = compile_data('./preprocessed_data')
    if not os.path.exists('./word_embeddings'):
        os.makedirs('./word_embeddings/figures')
        os.makedirs('./word_embeddings/models')
    list_of_words = [x for x in input('Please enter the list of words to plot separated by comas (", "): ').split(', ')]
    word_embeddings(df, list_of_words)
    return None


if __name__ == "__main__":
    import_tweets_and_do_embeddings()


