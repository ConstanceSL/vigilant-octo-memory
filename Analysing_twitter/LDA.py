import pandas as pd
from os import walk
from os.path import join
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

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
        df = df.dropna(subset=['text_lemmatized', 'orientation', 'date'])
        df['text_lemmatized'] = df['text_lemmatized'].astype(str).apply(sentence_to_list)
        data.append(df)
    return pd.concat(data).reset_index(drop=True)


"""
LDA
"""


def lda_crossvalidated(df, df_name, crossval, test_size, n_topics):
    """
    Fits the model for a specific data frame and number of topics
    :param df: Data frame on which to fit the model
    :param df_name: Name of the data frame
    :param crossval: Number of folds for cross validation
    :param test_size: Percentage of the data set to be kept for the test sample
    :param n_topics: Number of topics
    :return: A data frame with a summary of the model and its performance (one row per fold + one row of means)
    """
    topics = []
    train_coherence_cv = []
    train_coherence_umass = []
    test_perplexity = []
    for n in range(crossval):
        print('Fold {} of {}'.format(n + 1, crossval))
        # Splitting the texts between the test and train set
        train = list(np.random.choice(df['text_lemmatized'].tolist(),
                                      int(len(df['text_lemmatized'].tolist()) * (1 - test_size))))
        test = [tweet for tweet in df['text_lemmatized'].tolist() if tweet not in train]
        # Creating the corpus & dictionaries
        id2word_train = corpora.Dictionary(train)
        corpus_train = [id2word_train.doc2bow(text) for text in train]
        id2word_test = corpora.Dictionary(test)
        corpus_test = [id2word_train.doc2bow(text) for text in test]
        # Training the model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_train,
                                                    id2word=id2word_train,
                                                    num_topics=n_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        # Getting the topics
        topic = lda_model.print_topics()
        # Coherence on the train set (cv and umass)
        coherence_model_lda_cv = CoherenceModel(model=lda_model, texts=train, dictionary=id2word_train, coherence='c_v')
        coherence_cv = coherence_model_lda_cv.get_coherence()
        coherence_model_lda_umass = CoherenceModel(model=lda_model, texts=train, dictionary=id2word_train,
                                                   coherence='u_mass')
        coherence_umass = coherence_model_lda_umass.get_coherence()
        # Measuring perplexity on the test set
        perplexity = lda_model.log_perplexity(corpus_test)
        # Saving the results
        topics.append(topic)
        train_coherence_cv.append(coherence_cv)
        train_coherence_umass.append(coherence_umass)
        test_perplexity.append(perplexity)
    # Printing the average results of the folds
    print('Coherence (c_v): {}'.format(np.mean(train_coherence_cv)))
    print('Coherence (u_mass): {}'.format(np.mean(train_coherence_umass)))
    print('Perplexity: {}'.format(np.mean(test_perplexity)))
    # Saving the results
    topics.append('Means')
    train_coherence_cv.append(np.mean(train_coherence_cv))
    train_coherence_umass.append(np.mean(train_coherence_umass))
    test_perplexity.append(np.mean(test_perplexity))
    results = {'Data': [df_name] * (crossval + 1), 'Number_of_topics': [str(n_topics)] * (crossval + 1),
               'Topics': topics, 'Coherence_cv': train_coherence_cv, 'Coherence_umass': train_coherence_umass,
               'Perplexity': test_perplexity}
    return pd.DataFrame(results)[
        ['Data', 'Number_of_topics', 'Coherence_cv', 'Coherence_umass', 'Perplexity', 'Topics']]


def graph_results(df_results, df_name, list_of_number_of_topics):
    """
    Creates a figure summarising the results
    :param df_results: Data frame of results
    :param df_name: Name of the data frame (to save and label the results)
    :param list_of_number_of_topics: Number of topics (to label the results)
    :return: None
    """
    # Transforming the data frame of results in a format that can be plotted and keeping only the means
    df_results = df_results[df_results['Topics'] == 'Means']
    df_results = pd.melt(df_results, id_vars=['Number_of_topics'],
                         value_vars=['Coherence_cv', 'Coherence_umass', 'Perplexity'], var_name='Measure',
                         value_name='Score')
    # Plotting the figure
    fig, axis = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.subplots_adjust(wspace=0.2, hspace=0.9)
    sns.lineplot(x='Number_of_topics', y='Score', hue='Measure', data=df_results)
    title = 'LDA scores for ' + ' '.join(df_name.split('_'))
    fig.suptitle(title, fontsize=14)
    axis.set(ylabel='Scores', xlabel='Number of topics')
    #fig.autofmt_xdate()
    plt.savefig('./LDA/figures/' + df_name + '_' + str(min(list_of_number_of_topics)) + '_to_' +
                str(max(list_of_number_of_topics)) + '.pdf', bbox_inches='tight')
    plt.close()
    return None


def lda_topic_number(df, df_name, crossval, test_size, list_of_number_of_topics):
    """
    Runs the LDA_cv for different number of topics, and saves the results to a csv
    :param df: Data frame on which to fit the model
    :param df_name: Name of the data frame (used a name to save results
    :param crossval: Number of folds for cross validation
    :param test_size: Percentage of the data set to be kept for the test sample
    :param list_of_number_of_topics: List of numbers of topics for which to fit a model
    :return: Data frame of results
    """
    dfs_results = []
    for number_of_topics in list_of_number_of_topics:
        print('Fitting model for {} topics'.format(number_of_topics))
        dfs_results.append(lda_crossvalidated(df, df_name, crossval, test_size, number_of_topics))
    df_results = pd.concat(dfs_results).reset_index(drop=True)
    df_results.to_csv('./LDA/Results_' + df_name + '_' + str(min(list_of_number_of_topics)) + '_to_' +
                str(max(list_of_number_of_topics)) + '.csv')
    graph_results(df_results, df_name, list_of_number_of_topics)
    return df_results


def batch_lda(list_of_dfs, list_of_dfs_names, crossval, test_size, list_of_number_of_topics):
    """
    Runs the LDA for all the data frames and all the number of topics
    :param list_of_dfs: List of data frames on which to fit models
    :param list_of_dfs_names: List of names for the data frame (to label the results)
    :param crossval: Number of folds for the cross validation
    :param test_size: Proportion of the data to be kept for testing purposes
    :param list_of_number_of_topics: List of number of topics for which to fit a model
    :return: A data frame with all the results
    """
    results = []
    for n, df, df_name in zip(range(len(list_of_dfs)), list_of_dfs, list_of_dfs_names):
        print('-' * 100 + '\nLDA for dataframe {} ({} of {})'.format(df_name, n + 1, len(list_of_dfs)))
        results.append(lda_topic_number(df, df_name, crossval, test_size, list_of_number_of_topics))
    df_results = pd.concat(results).reset_index(drop=True)
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    df_results.to_csv('./LDA/Results_batch_' + date + '.csv')
    return df_results


"""
GENERAL FUNCTION
"""


def lda():
    """
    General function that loads the data and fits the LDA models
    Prompts the user to enter the path where the project is located
    Prompts the user to enter the number of topics to be fitted
    :return: None
    """
    path = input('Please enter the path of the project directory: ')
    os.chdir(path)
    df = compile_data(path + '/preprocessed_data')
    if not os.path.exists('./LDA'):
        os.makedirs('./LDA/figures/')
    print('Data loaded')
    topic_numbers = [int(x) for x in input('Please enter the start, end, and step for the range of number of topics ('
                                           'separated by ", "): ').split(', ')]
    list_of_number_of_topics = list(range(topic_numbers[0], topic_numbers[1], topic_numbers[2]))
    crossval = int(input('Please enter the number of folds for the cross validation: '))
    test_size = float(input('Please enter the size of the test set (between 0 and 1): '))
    list_of_dfs = []
    list_of_dfs_names = []
    for orientation in set(df['orientation']):
        if str(orientation) == 'nan' or str(orientation) == 'None':
            pass
        else:
            df_orient = df[df['orientation'] == orientation]
            list_of_dfs.append(df_orient)
            list_of_dfs_names.append(str(orientation) + '_tweets_all_dates')
            for date in set(df['date']):
                list_of_dfs.append(df_orient[df_orient['date'] == date])
                list_of_dfs_names.append(str(orientation) + '_tweets_' + str(date))
    batch_lda(list_of_dfs, list_of_dfs_names, crossval, test_size, list_of_number_of_topics)
    return None


if __name__ == "__main__":
    lda()
