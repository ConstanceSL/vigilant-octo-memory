import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from os import walk
from os.path import join
import os

"""
IMPORTING THE DATA
"""


def string_to_list(column):
    """
    Many of the cells in the data frame contain lists that are converted to string when saved to csv
    This converts them back to lists
    :param column: column to convert from string to list
    :return: list
    """
    try:
        col = re.sub("\[|\]|'", '', column)
        col = col.split(', ')
    except Exception:
        col = None
    return col


def import_data(path):
    """
    Imports the data for one path, corrects formatting issues, and adds some useful columns for the stats
    :param path: path for one csv file
    :return: data frame
    """
    try:
        df = pd.read_csv(path, engine='c')
    except Exception:
        df = pd.read_csv(path, engine='python')
    # Cleaning up the results and saving
    dates = [date for date in list(set(df['date'])) if type(date) == str]
    dates = [date for date in dates if len(date) == 10]
    df = df[df['date'].isin(dates)]
    df.fillna(value=np.nan, inplace=True)
    df = df.dropna(subset=['text', 'id_number', 'text_lemmatized'])
    # Changing the format of some variables
    df['text_range'] = df['text_range'].astype(str).apply(string_to_list)
    df['hashtags'] = df['hashtags'].astype(str).apply(string_to_list)
    df['reply_to_user_id'] = df['reply_to_user_id'].astype(str).apply(string_to_list)
    df['mentions_names'] = df['mentions_names'].astype(str).apply(string_to_list)
    df['links'] = df['links'].astype(str).apply(string_to_list)
    df['media_types'] = df['media_types'].astype(str).apply(string_to_list)
    # Adding variables
    df = df.assign(media_number=[len(media) if media[0] != 'nan' else 0 for media in df['media_types'].tolist()])
    df = df.assign(has_media=[1 if media != 0 else 0 for media in df['media_number'].tolist()])
    df = df.assign(has_mention=[1 if len(mentions) != 0 else 0 for mentions in df['mentions_names'].tolist()])
    df = df.assign(mentions_number=[len(mentions) for mentions in df['mentions_names'].tolist()])
    df = df.assign(is_reply=[1 if reply[0] != 'nan' else 0 for reply in df['reply_to_user_id'].tolist()])
    df = df.assign(links_number=[len(links) if links[0] != 'nan' else 0 for links in df['links'].tolist()])
    df = df.assign(has_link=[1 if links != 0 else 0 for links in df['links_number'].tolist()])
    df = df.assign(is_verified=[1 if verif == True else 0 for verif in df['user_verified'].tolist()])
    df = df.assign(is_sensitive=[1 if sensi == True else 0 for sensi in df['possibly_sensitive'].tolist()])
    df = df.assign(is_quote=[1 if quote == True else 0 for quote in df['is_quote_status'].tolist()])
    df = df.assign(text_length=[int(span[1]) if len(span) == 2 else None for span in df['text_range'].tolist()])
    df = df.assign(hashtags_number=[len(tags) for tags in df['hashtags'].tolist()])
    df = df.assign(retweets_per_100_followers=[retweets * 100 / (followers + 0.0000001) for retweets, followers in
                                               zip(df['retweet_count'].tolist(), df['user_followers_count'].tolist())])
    df = df.assign(lang_user_en=[1 if lang == 'en' else 0 for lang in df['user_lang'].tolist()])
    return df.reset_index(drop=True)


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
        data.append(import_data(path))
    return pd.concat(data).reset_index(drop=True)


"""
TABLES 
"""


def percentages(df, group_by):
    """
    Gets stats for binary variables (e.g., the tweet is a reply to another tweet or not)
    :param df: data
    :param group_by: what to group the data by ('query', 'orientation' or 'date')
    :return: a data frame of percentages
    """
    count_tweets = df.groupby(group_by)['text'].count()
    count_replies = df.groupby(group_by)['is_reply'].sum()
    count_media = df.groupby(group_by)['has_media'].sum()
    count_mentions = df.groupby(group_by)['has_mention'].sum()
    count_sensitive = df.groupby(group_by)['is_sensitive'].sum()
    count_quote = df.groupby(group_by)['is_quote'].sum()
    count_verified = df.groupby(group_by)['is_verified'].sum()
    count_links = df.groupby(group_by)['has_link'].sum()
    count_english = df.groupby(group_by)['lang_user_en'].sum()
    df_count = pd.DataFrame(dict(Tweets=count_tweets, Replies=count_replies,
                                 Media=count_media, Mention=count_mentions,
                                 Sensitive=count_sensitive, Quote=count_quote,
                                 Verified=count_verified,
                                 Links=count_links, English=count_english))[['Tweets', 'Replies',
                                                                             'Quote', 'Mention',
                                                                             'Links', 'Media',
                                                                             'Sensitive', 'Verified',
                                                                             'English']]
    df_percentage = df_count.astype('float64')
    df_percentage["Percentage"] = np.nan
    df_percentage = df_percentage.append(
        pd.DataFrame(df_percentage.sum()).transpose(copy=True).rename(index={0: 'Totals'}))
    total_tweets = df_percentage['Tweets']['Totals']
    for index, row in df_percentage.iterrows():
        row['Replies'] = row['Replies'] * 100 / row['Tweets']
        row['Quote'] = row['Quote'] * 100 / row['Tweets']
        row['Mention'] = row['Mention'] * 100 / row['Tweets']
        row['Links'] = row['Links'] * 100 / row['Tweets']
        row['Media'] = row['Media'] * 100 / row['Tweets']
        row['Sensitive'] = row['Sensitive'] * 100 / row['Tweets']
        row['Verified'] = row['Verified'] * 100 / row['Tweets']
        row['English'] = row['English'] * 100 / row['Tweets']
        row['Percentage'] = row['Tweets'] * 100 / total_tweets
    return df_percentage


def get_means_df(df, group_by):
    """
    Gets average for numerical variables
    :param df: data
    :param group_by: what to group the data by ('query', 'orientation' or 'date')
    :return: dataframe of means
    """
    mean_length = df.groupby(group_by)['text_length'].mean()
    mean_retweets = df.groupby(group_by)['retweet_count'].mean()
    mean_favorites = df.groupby(group_by)['favorites_count'].mean()
    mean_hashtags = df.groupby(group_by)['hashtags_number'].mean()
    mean_mentions = df.groupby(group_by)['mentions_number'].mean()
    mean_links = df.groupby(group_by)['links_number'].mean()
    mean_user_statuses = df.groupby(group_by)['user_statuses_count'].mean()
    mean_followers = df.groupby(group_by)['user_followers_count'].mean()
    mean_totals = pd.DataFrame(df[['text_length', 'retweet_count', 'favorites_count',
                                   'hashtags_number', 'mentions_number', 'links_number',
                                   'user_statuses_count', 'user_followers_count']].mean()).transpose(copy=True).rename(
        index={0: 'Average'})
    mean_totals.columns = ['Length', 'Retweets', 'Favorites', 'Hashtags', 'Mentions', 'Links', 'Statuses', 'Followers']

    df_mean = pd.DataFrame(dict(Length=mean_length, Retweets=mean_retweets,
                                Favorites=mean_favorites, Hashtags=mean_hashtags,
                                Mentions=mean_mentions, Links=mean_links,
                                Statuses=mean_user_statuses, Followers=mean_followers))
    df_mean = df_mean.append(mean_totals)
    return df_mean


def analysis(df, folder_path):
    """
    Runs the statistical analyses and saves them to different files
    :param df: data
    :param folder_path: folder where to save the results
    :return: None
    """
    for group_by in [['query'], ['orientation'], ['date'], ['query', 'date'], ['orientation', 'date']]:
        percentages(df, group_by).to_csv(folder_path + '/stats_percentage_' + '_'.join(group_by) + '.csv')
        get_means_df(df, group_by).to_csv(folder_path + '/stats_means_' + '_'.join(group_by) + '.csv')
    print('Analyses saved to file')
    return None


"""
FIGURES
"""


def make_figure(df, folder_path, column, ylab, title, bars, colours, extension):
    """
    Makes and saves barplots
    :param df: data
    :param folder_path: directory where to save the figures
    :param column: variable to plot
    :param ylab: label for the y axis
    :param title: title of the plot
    :param bars: variable to group in the bars
    :param colours: Variable to group the bars (and colour them)
    :param extension: end part of the file name
    :return: None, but saves the figure to file
    """
    fig, axis = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.subplots_adjust(wspace=0.2, hspace=0.9)
    sns.barplot(x=bars, y=column, hue=colours, data=df, ax=axis)
    if bars == 'query':
        axis.set(ylabel=ylab, xlabel='Hashtag')
    if bars == 'orientation':
        axis.set(ylabel=ylab, xlabel='Orientation')
    fig.suptitle(title, fontsize=14)
    fig.autofmt_xdate()
    # fig.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.1, title=colours)
    plt.savefig(folder_path + '/' + column + extension, bbox_inches='tight')
    plt.close()
    return None


def figures(df, folder_path):
    """
    Creates all the figures and combinations
    :param df: data
    :param folder_path: folder where to save the figures
    :return: None
    """
    columns = ['text_length', 'hashtags_number', 'is_reply', 'has_mention', 'mentions_number',
               'has_link', 'links_number', 'has_media', 'is_quote', 'retweet_count', 'favorites_count',
               'user_followers_count', 'is_verified', 'is_sensitive', 'retweets_per_100_followers', 'lang_user_en',
               'user_statuses_count']
    ylabels = ['Length', 'Number of hashtags', 'Replies', 'Mentions', 'Number of mentions',
               'Link', 'Number of links', 'Media', 'Quote', 'Retweets', 'Favorites',
               'Followers', 'Verified', 'Sensitive', 'Retweets per 100 followers', 'Accounts in English',
               'Number of statuses']
    titles = ['Average length of tweet', 'Average number of hashtags per tweet',
              'Proportion of tweets that are replies',
              'Proportion of tweets with mentions', 'Average number of mentions', 'Proportion of tweets with links',
              'Average number of links per tweet', 'Proportion of tweets with media',
              'Proportion  of tweets with quote',
              'Average number of retweets per tweet', 'Average number of favorites per tweet',
              'Average number of followers per tweet author', 'Proportion of verified user accounts',
              'Proportion of tweets classified sensitive', 'Average number of retweets per 100 followers',
              'Proportion of accounts in English', 'Average number of statuses per user']
    for column, ylab, title in zip(columns, ylabels, titles):
        # Grouped first by hashtag, then by orientation
        make_figure(df.sort_values(['orientation', 'query']), folder_path, column, ylab, title,
                    bars='query', colours='orientation', extension='_by_tag_orientation.pdf')
        # Grouped first by hashtag, then by date
        make_figure(df.sort_values(['orientation', 'query', 'date'], ascending=[1, 1, 0]), folder_path, column, ylab,
                    title, bars='query', colours='date', extension='_by_tag_date.pdf')
        # Grouped first by orientation, then by date
        make_figure(df.sort_values(['orientation', 'query', 'date'], ascending=[1, 1, 0]), folder_path, column, ylab,
                    title, bars='orientation', colours='date', extension='_by_orientation_date.pdf')
    print('Figures saved to file')
    return None


"""
WORD FREQUENCIES
"""


def word_frequency(df, folder_path):
    """
    Looks at word frequencies for side of the debate, then each hashtag, first for all dates together, then per date
    :param folder_path: where to save the results
    :param df: data
    :return: a data frame of word frequencies, also saved to file
    """
    # Looking at the frequencies grouped by orientation
    dfs = []
    for orientation in set(df['orientation'].tolist()):
        df_orient = df[df['orientation'] == orientation]
        name = orientation + '_all_dates'
        tweets = [tweet for tweet in df_orient['text_lemmatized'].tolist() if type(tweet)==str]
        words = [word for tweet in tweets for word in tweet.split(' ')]
        #words = [word for tweet in df_orient['text_lemmatized'].tolist() for word in tweet.split(' ') if
        #         type(tweet) == str]
        fdist = nltk.FreqDist(words)
        words = []
        frequencies = []
        percentages = []
        for word, frequency in fdist.most_common(10000):
            words.append(word)
            frequencies.append(frequency)
            percentages.append(frequency / df_orient.shape[0])
        results = {'Words_' + name: words, 'Frequencies_' + name: frequencies, 'Average_per_tweet_' + name: percentages}
        df_results = pd.DataFrame(results)[['Words_' + name, 'Frequencies_' + name, 'Average_per_tweet_' + name]]
        df_results.index += 1
        dfs.append(df_results)
        for date in set(df_orient['date'].tolist()):
            df_date = df_orient[df_orient['date'] == date]
            name = orientation + '_' + date
            tweets = [tweet for tweet in df_date['text_lemmatized'].tolist() if type(tweet) == str]
            words = [word for tweet in tweets for word in tweet.split(' ')]
            #words = [word for tweet in df_date['text_lemmatized'].tolist() for word in tweet.split(' ') if
            #         type(tweet) == str]
            fdist = nltk.FreqDist(words)
            words = []
            frequencies = []
            percentages = []
            for word, frequency in fdist.most_common(10000):
                words.append(word)
                frequencies.append(frequency)
                percentages.append(frequency / df_date.shape[0])
            results = {'Words_' + name: words, 'Frequencies_' + name: frequencies,
                       'Average_per_tweet_' + name: percentages}
            df_results = pd.DataFrame(results)[['Words_' + name, 'Frequencies_' + name, 'Average_per_tweet_' + name]]
            df_results.index += 1
            dfs.append(df_results)
    # Looking at the frequencies grouped by side of the debate
    for hashtag in set(df['query'].tolist()):
        df_tag = df[df['query'] == hashtag]
        name = hashtag + '_all_dates'
        tweets = [tweet for tweet in df_tag['text_lemmatized'].tolist() if type(tweet) == str]
        words = [word for tweet in tweets for word in tweet.split(' ')]
        #words = [word for tweet in df_tag['text_lemmatized'].tolist() for word in tweet.split(' ') if
        #         type(tweet) == str]
        fdist = nltk.FreqDist(words)
        words = []
        frequencies = []
        percentages = []
        for word, frequency in fdist.most_common(10000):
            words.append(word)
            frequencies.append(frequency)
            percentages.append(frequency / df_tag.shape[0])
        results = {'Words_' + name: words, 'Frequencies_' + name: frequencies, 'Average_per_tweet_' + name: percentages}
        df_results = pd.DataFrame(results)[['Words_' + name, 'Frequencies_' + name, 'Average_per_tweet_' + name]]
        df_results.index += 1
        dfs.append(df_results)
        for date in set(df_tag['date'].tolist()):
            df_date = df_tag[df_tag['date'] == date]
            name = hashtag + '_' + date
            tweets = [tweet for tweet in df_date['text_lemmatized'].tolist() if type(tweet) == str]
            words = [word for tweet in tweets for word in tweet.split(' ')]
            #words = [word for tweet in df_date['text_lemmatized'].tolist() for word in tweet.split(' ') if
            #         type(tweet) == str]
            fdist = nltk.FreqDist(words)
            words = []
            frequencies = []
            percentages = []
            for word, frequency in fdist.most_common(10000):
                words.append(word)
                frequencies.append(frequency)
                percentages.append(frequency / df_date.shape[0])
            results = {'Words_' + name: words, 'Frequencies_' + name: frequencies,
                       'Average_per_tweet_' + name: percentages}
            df_results = pd.DataFrame(results)[['Words_' + name, 'Frequencies_' + name, 'Average_per_tweet_' + name]]
            df_results.index += 1
            dfs.append(df_results)
    result_df = pd.concat(dfs, axis=1)
    result_df.to_csv(folder_path + '/word_frequencies.csv')
    print('Word frequencies saved to file')
    return result_df


"""
GENERAL FUNCTION
"""


def import_and_analyse_tweets():
    """
    General function that loads the data, analyses it, and saves the results
    Prompts the user to enter the path to the directory where the data is stored and of where to save the results
    :return: None
    """
    path = input('Please enter the path of the project directory: ')
    os.chdir(path)
    df = compile_data(path + '/preprocessed_data')
    if not os.path.exists('./stats'):
        os.makedirs('./stats/tables')
        os.makedirs('./stats/figures')
        os.makedirs('./stats/word_frequencies')
    print('Running statistical analyses')
    analysis(df, './stats/tables')
    print('Creating the figures')
    figures(df, './stats/figures')
    print('Analysing word frenquencies')
    word_frequency(df, './stats/word_frequencies')
    print('All analyses completed')
    return None


if __name__ == "__main__":
    import_and_analyse_tweets()
