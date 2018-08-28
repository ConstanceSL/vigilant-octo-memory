import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
import gensim
import spacy
import os
import numpy as np


"""
IMPORTING THE DATA
"""


def string_to_list(column):
    """
    Many of the cells in the dataframe contain lists that are converted to string when saved to csv
    This converts them back to lists
    :param column: column to convert from string to list
    :return: list
    """
    col = re.sub("\[|\]|'", '', column)
    col = col.split(', ')
    return col


def import_data(path):
    """
    Imports the csv to a data frame, and fixes the columns that contain lists (converting them from string to list)
    :param path: path where the csv was saved
    :return: data frame of tweets
    """
    df_original = pd.read_csv(path, index_col=0)
    # Getting the 'name' of the data frame
    name = df_original['query'].tolist()[0] + '_' + df_original['date'].tolist()[0]
    # Dropping duplicates and empty tweets
    df = df_original.drop_duplicates(subset='id_number')
    df.fillna(value=np.nan, inplace=True)
    df.dropna(subset=['text', 'id_number', 'date'])
    if df.shape[0] != df_original.shape[0]:
        print('Dropped {} duplicates and empty tweets for {}'.format(df_original.shape[0] - df.shape[0], name))
    # Fixing empty columns and lists turned to strings
    df['text_range'] = df['text_range'].astype(str).apply(string_to_list)
    df['hashtags'] = df['hashtags'].astype(str).apply(string_to_list)
    df['mentions_screen_names'] = df['mentions_screen_names'].astype(str).apply(string_to_list)
    df['mentions_names'] = df['mentions_names'].astype(str).apply(string_to_list)
    df['links'] = df['links'].astype(str).apply(string_to_list)
    df['media_urls'] = df['media_urls'].astype(str).apply(string_to_list)
    return df


"""
CLEANING UP THE HASHTAGS
"""


def get_hashtags(df):
    """
    Creates a list of hashtags that can be split based on the use of caps
    Prints the percentage of hashtags that can be 'solved' (turned into text) this way, and whether
    some unsolved hashtags appear more than once (so if they should be solved by hand)
    :param df: data frame of tweets
    :return: list of solved tags
    """
    # Making list of all hashtags, removing the ones that are only 1 letter long
    tags = set([hashtag for hashtags in df['hashtags'] for hashtag in hashtags if len(hashtag) > 1])
    print('Collected {} unique hashtags'.format(len(tags)))
    # Selecting the tags that are either easier to split
    upper_case_tags = [tag for tag in tags if tag[0].isupper() and tag[1].islower()]
    # Splitting the tags on uppercase letters or on the numbers
    split_upper_tags = [' '.join(re.findall('[A-Z][a-z]*|[0-9]+', tag)) for tag in upper_case_tags]
    # Tags starting with a number
    number_tags = set([tag for tag in tags if tag[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']])
    split_number_tags = [' '.join(re.findall('[A-Z][a-z]*|[a-z]+|[0-9]+', tag)) for tag in number_tags]
    # Merging the lists
    split_tags = sorted(split_upper_tags + split_number_tags)
    # Removing the annoying empty tag
    split_tags = [tag for tag in split_tags if tag != '']
    ## Separating the tags between the ones that follow a tag like format and the others
    # First, formatting the numbers better
    formatted_tags = [re.sub(' [sS][tT] ', 'st ', tag) for tag in split_tags]
    formatted_tags = [re.sub(' [nN][dD] ', 'nd ', tag) for tag in formatted_tags]
    formatted_tags = [re.sub(' [rR][dD] ', 'rd ', tag) for tag in formatted_tags]
    formatted_tags = [re.sub(' [tT][hH] ', 'th ', tag) for tag in formatted_tags]
    formatted_tags = [re.sub(' [sS][tT]$', 'st', tag) for tag in formatted_tags]
    formatted_tags = [re.sub(' [nN][dD]$', 'nd', tag) for tag in formatted_tags]
    formatted_tags = [re.sub(' [rR][dD]$', 'rd', tag) for tag in formatted_tags]
    formatted_tags = [re.sub(' [tT][hH]$', 'th', tag) for tag in formatted_tags]
    # keeping only the tags that have the proper shape, first removing those with words that don't start with an uppercase
    proper_tags = [tag for tag in formatted_tags if
                  re.match('([A-Z][a-z]*|[0-9]+[a-z]*)( [A-Z][a-z]*| [0-9]+[a-z]*)*', tag).span()[1] == len(tag)]
    # Removing the tags that have too many uppercases
    proper_tags = set([tag for tag in proper_tags if re.search('.* [A-Z] [A-Z] .*', tag) is None])
    ## Removing the list of cleaned tags from the original list of tags, to estimate how many tags won't be matched
    # Removing spaces and uppercases
    lowercase_proper_tags = [re.sub(' ', '', tag.lower()) for tag in proper_tags]
    tags_unmatched = [tag for tag in tags if tag.lower() not in lowercase_proper_tags]
    # Printing info
    percentage_unmatched = int(len(set(tags_unmatched)) * 100 / len(set(tags)))
    print('Approximatively {}% tags solved'.format(100 - percentage_unmatched))
    df_tags = pd.DataFrame.from_dict(Counter(tags_unmatched), orient='index').sort_values([0],
                                                                                          ascending=False).reset_index(
        inplace=False).rename(columns={'index': 'Hashtag', 0: 'Count'})
    frequent_unmatched_tags = df_tags[df_tags['Count'] > 1]
    if frequent_unmatched_tags.shape[0] == 0:
        print('All unsolved tags appear only once in the data')
    else:
        print('The following unsolved tags appeared more than once:')
        print(frequent_unmatched_tags)
    return [tag.lower() for tag in proper_tags]


def corrected_tag_list(proper_tags):
    """
    Prompts the user to enter a correction for the tags that have more than one solution
    Saves the final list of corrected tags in a location specified by the user
    :param proper_tags: Automatically corrected tags of the get_hashtags function
    :return: list of automatically and manually corrected tags
    """
    # Getting the tags that ended up with different separations, to manually correct them
    df_tags = pd.DataFrame.from_dict(Counter([re.sub(' ', '', tag.lower()) for tag in proper_tags]), orient='index') \
        .sort_values([0], ascending=False).reset_index(inplace=False).rename(columns={'index': 'Hashtag', 0: 'Count'})
    tags_to_correct = df_tags[df_tags['Count'] > 1]['Hashtag'].tolist()
    tags_ok = df_tags[df_tags['Count']==1]['Hashtag'].tolist()
    # Checking if a list of corrections already exists, and removing already corrected tags from the list
    filepath = 'corrected_hashtags.csv'
    if os.path.exists(filepath):
        already_corrected_tags = [re.sub(' ', '', tag.lower()) for tag in
                                  pd.read_csv(filepath, header=None)[0].tolist()]
        tags_to_correct = [tag for tag in tags_to_correct if tag not in already_corrected_tags]
        tags_ok = [tag for tag in tags_ok if tag not in already_corrected_tags]
    # Listing the tags that should be corrected
    if len(tags_to_correct) != 0:
        print('The following tags have more than one solution and need to be manually corrected: ')
        print(tags_to_correct)
        # Getting the manual correction
        manually_corrected_tags = [x for x in input('Please enter the list of corrected tags separated by comas (", "): ')
            .split(', ')]
    else:
        manually_corrected_tags = []
    # Replacing the ok tags by their split/corrected version
    tags_ok = [tag for tag in proper_tags if re.sub(' ', '', tag.lower()) in tags_ok]
    # Putting all the lists of tags together and making sure there are no repetitions
    tags_cleaned = set(tags_ok + manually_corrected_tags)
    # Save the list of tags
    file = open(filepath, 'a+')
    file.writelines("%s\n" % item for item in tags_cleaned)
    return tags_cleaned


"""
FORMATTING THE TWEETS
"""


def formatting_text(row, tags_dict):
    """
    Function to be applied to the dataframe of tweets (in the formatting_tweets function below)
    :param row: Row in the data frame of tweets
    :param tags_dict: Dictionnary of tags and their correction
    :return: formatted text of the tweet
    """
    text = row['text'].lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('&amp;', 'and', text)
    text = re.sub(' https:[^\s]*', ' ', text)
    # removing mentions and replacing them by the user names
    mentions_dict = {'@' + key.lower(): value for (key, value) in zip(row['mentions_screen_names'],
                                                                      row['mentions_names'])}
    for pseudo, name in mentions_dict.items():
        text = text.replace(pseudo, name)
    # Removing emojis
    text = ''.join(c for c in text if c <= '\uFFFF')
    text = re.sub('❤️', '', text)
    # Removing tags: replacing the formatted ones
    hashtags = row['hashtags']
    local_tags_dict = dict()
    for hashtag in hashtags:
        hashtag = '#' + hashtag.lower()
        try:
            local_tags_dict[hashtag] = tags_dict[hashtag]
        except Exception:
            pass
    for tag, cleaned_text in local_tags_dict.items():
        text = re.sub(tag + ' ', cleaned_text + ' ', text)
        text = re.sub(tag + '$', cleaned_text, text)
    # Removing punctuation and extra spaces, putting everything to lower case to finish
    text = re.sub('[\.\+"\*\%\&\/\(\)\=\?\`\^\!\;\,\:\_\°\<\>\~]', ' ', text)
    text = re.sub("[\’\'\-]", '', text)
    # Deleting remaining tags
    text = re.sub('#[0-9a-z]*? |#[0-9a-z]*?$', ' ', text)
    text = re.sub('[ ]{2,}', ' ', text)
    text = text.lower()
    return text


def tokenisation_stopwords(text):
    """
    Tokenising and removing stopwords in the formatted text
    :param text: formatted text from the formatting_text function output
    :return: Tokenised text without the stopwords
    """
    text = text.split(' ')
    # Removing 1 letter long words and empty strings, and the remaining bad characters
    text = [re.sub('[^a-zA-Z0-9]*|[\]{1,2}u[a-z0-9]*', '', word) for word in text if len(word) > 1]
    # Removing stopwords
    stop_words_extra =['aint', 'arent', 'cant', 'couldve', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt',
                       'havent', 'hed', 'hell', 'hes', 'heve', 'Id', 'Ill', 'Im', 'Ive', 'isnt', 'itll', 'its',
                       'lets', 'mustnt', 'shed', 'shell', 'shes', 'shouldve', 'shouldnt', 'thatll', 'thats',
                       'theres', 'theyd', 'theyll', 'theyre', 'theyve', 'wasnt', 'wed', 'well', 'were', 'weve',
                       'werent', 'whats', 'wheres', 'wholl', 'whos', 'wont', 'wouldve', 'wouldnt', 'yall', 'youd',
                       'youll','youre', 'youve']
    text = [word for word in text if word not in stopwords.words('english')]
    text = [word for word in text if word not in stop_words_extra]
    return text


def formatting_tweets(df, filepath):
    """
    Formats the tweets in the dataframe uploaded by import_data
    :param df: dataframe of tweets
    :param tags_cleaned: output of corrected_tag_list
    :param filepath: where to save the formatted tweets
    :return: dataframe of tweets with an extra column of formatted tweets
    """
    total_to_format = df.shape[0]
    print('{} tweets to format'.format(total_to_format))
    n = 0
    m = 0
    tenth = int(total_to_format / 10)
    # Creating a dictionnary of tags to replace the values
    tags_cleaned = pd.read_csv('corrected_hashtags.csv', header=None)[0].tolist()
    tags_condensed = ['#' + re.sub(' ', '', tag.lower()) for tag in tags_cleaned]
    tags_dict = {key: value for (key, value) in zip(tags_condensed, tags_cleaned)}
    # Replacing the tags and removing/correcting basic formatting, including emoji and URLs
    formatted_text = []
    tokenised_text = []
    for index, row in df.iterrows():
        try:
            text = formatting_text(row, tags_dict)
            formatted_text.append(text)
            tokenised_text.append(tokenisation_stopwords(text))
        except Exception:
            formatted_text.append('')
            tokenised_text.append('')
        n += 1
        if n == tenth:
            m += 1
            n = 0
            print('{}% formatted'.format(m * 10))
    df = df.assign(text_formatted = formatted_text)
    df = df.assign(text_tokenised = tokenised_text)
    if filepath != None:
        df.to_csv(filepath)
        print('Results saved to file')
    print('All tweet formatted and tokenised')
    return df


"""
LEMMATISATION
(large chunks taken from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) 
"""


def make_bigrams(tweets, bigram_mod):
    """
    Creates a bigrams (grouping words frequently occuring together)
    :param tweets: list of tweets
    :param bigram_mod: gensim bigram model (created in the lemmatisation function below)
    :return: tokenised tweets with bigrams
    """
    return [bigram_mod[doc] for doc in tweets]


def make_trigrams(tweets, bigram_mod, trigram_mod):
    """
    Same with trigrams
    """
    return [trigram_mod[bigram_mod[doc]] for doc in tweets]


def lemmatisation(df, filepath):
    """
    Lemmatises the tweets and creates bigrams and trigrams of frequently occurring expressions
    :param df: formated dataframe from previous step
    :param filepath: where to save the results
    :return: dataframe with two new columns: lemmatised tweets with and without the hashtag in the original query
    """
    total_to_format = df.shape[0]
    print('{} tweets to format'.format(total_to_format))
    n = 0
    m = 0
    tenth = int(total_to_format / 10)

    # Loading Spacy lemmatiser
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Creates lists for the outputs, one including the hashtag from the original query, one ignoring it
    tweets_out = []
    tweets_out_without_tag = []
    hashtag = df['query'].tolist()[0]
    tweets = df['text_tokenised'].tolist()

    # Creating the bigram and trigram models, then applying them
    bigram = gensim.models.Phrases(tweets, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[tweets], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    tweets = make_trigrams(tweets, bigram_mod, trigram_mod)

    # To keep only nouns, adjectives, verbs, adverbs, foreign words, proper nouns
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV', 'X', 'PROPN']
    for tweet in tweets:
        clean_tweet = []
        clean_tweet_without_tag = []
        for word in tweet:
            # Remove the remaining bad characters
            word = re.sub('[^a-zA-Z]*|[\]{1,2}u[a-z0-9]*', '', word)
            clean_tweet.append(word)
            if word not in hashtag:
                clean_tweet_without_tag.append(word)
        doc = nlp(" ".join(clean_tweet))
        tweets_out.append(" ".join(
            [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        doc_without_tag = nlp(" ".join(clean_tweet_without_tag))
        tweets_out_without_tag.append(" ".join(
            [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc_without_tag if
             token.pos_ in allowed_postags]))
        n += 1
        if n == tenth:
            m += 1
            n = 0
            print('{}% formatted'.format(m * 10))
    df = df.assign(text_lemmatized=tweets_out)
    df = df.assign(text_lemmatized_without_tag=tweets_out_without_tag)
    # Cleaning up the results and saving
    dates = [date for date in list(set(df['date'])) if type(date) == str]
    dates = [date for date in dates if len(date) == 10]
    df = df[df['date'].isin(dates)]
    df.fillna(value=np.nan, inplace=True)
    df = df.dropna(subset=['text_lemmatized', 'orientation', 'date', 'query'])
    df.to_csv(filepath)
    print('Results saved to file')
    print('All tweets lemmatised')
    return df


"""
GENERAL FUNCTION
"""


def import_and_format_tweets():
    """
    General function that loads the data, formats it, and saves it
    Prompts the user to enter the path where the project is located
    Prompts the user to enter a cleaned version of the tags that couldn't automatically be corrected
    :return: data frame of formatted tweets
    """
    path = input('Please enter the path of the project directory: ')
    os.chdir(path)
    # Listing the data to process
    files_to_process = os.listdir('./raw_data')
    # Checking if some of the data has been processed already
    if os.path.exists('./preprocessed_data'):
        processed_files = os.listdir('./preprocessed_data')
        files_to_process = [file for file in files_to_process if file not in processed_files]
    # Creating the folder of processed data if it doesn't exist
    if not os.path.exists('./preprocessed_data'):
        os.makedirs('./preprocessed_data')
    # Processing the data
    print('{} files to process'.format(len(files_to_process)))
    for file in files_to_process:
        print('-' * 100 + '\nProcessing file {}'.format(file))
        df = import_data('./raw_data/' + file)
        print('Solving hashtags')
        proper_tags = get_hashtags(df)
        tags_cleaned = corrected_tag_list(proper_tags)
        print('Formatting and tokenising the tweets')
        df = formatting_tweets(df, './preprocessed_data/' + file)
        print('Lemmatising the tweets')
        lemmatisation(df, './preprocessed_data/' + file)
    print('All files processed')
    return None


if __name__ == "__main__":
    import_and_format_tweets()
