import tweepy
import pandas as pd
import datetime
import os
import numpy as np

# Importing the keys for the twitter API
keys = pd.read_csv('/Users/Constance/keys/twitter_keys.csv', index_col=0)

consumer_key = keys.loc['consumer_key'][0]
consumer_secret = keys.loc['consumer_secret'][0]
access_token = keys.loc['access_token'][0]
access_token_secret = keys.loc['access_token_secret'][0]


# Defining the main functions
def download_tweets(query, limit):
    """
    Downloads the tweets in the query. Because of the limit on the number of requests,
    can take a while. Doesn't include retweets, and has a random old date for the time limit
    (will get only 7 days of data anyway)
    :param query: hashtag to search
    :param limit: max number of tweets to download (it tends to crash above 100'000 tweets)
    :return: list of tweets (json format)
    """
    tweets = []
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    n = 0
    m = 0
    print('-' * 100 + '\nStarting download for ' + query)
    for tweet in tweepy.Cursor(api.search, q=query + ' -filter:retweets', count=100,
                               lang="en",
                               since="2017-06-28",
                               include_rts=False,
                               tweet_mode="extended").items():
        tweets.append(tweet._json)
        n += 1
        if n == 500:
            print('Tweets {}-{} downloaded'.format(m + n - 499, m + n))
            n = 0
            m += 500
        if m + n == limit:
            break
    print('Downloaded {} tweets'.format(m + n))
    return tweets


def format_tweets(tweets, query, orientation):
    """
    Formats the downloaded tweets into lists (to be converted to a dataframe), keeping only the relevant info.
    Not all fields exist in the json, so uses a lot of try/except to catch this and puts 'None' for the exceptions
    :param tweets: list of tweets (json format)
    :return: list of tweets (list format) with only the relevant fields
    """
    formatted_tweets = []
    for tweet in tweets:
        formatted_tweet = []
        formatted_tweet.extend([tweet['created_at'], tweet['id'], tweet['full_text'],
                                tweet['truncated'], tweet['display_text_range']])
        try:
            formatted_tweet.append([tag['text'] for tag in tweet['entities']['hashtags']])
        except Exception:
            formatted_tweet.append(np.nan)
        try:
            formatted_tweet.extend([[mention['screen_name'] for mention in tweet['entities']['user_mentions']],
                                    [mention['name'] for mention in tweet['entities']['user_mentions']]])
        except Exception:
            formatted_tweet.extend([np.nan, np.nan])
        try:
            formatted_tweet.append([link['url'] for link in tweet['entities']['urls']])
        except Exception:
            formatted_tweet.append(np.nan)
        try:
            formatted_tweet.extend([[media['media_url'] for media in tweet['entities']['media']],
                                    [media['type'] for media in tweet['entities']['media']]])
        except Exception:
            formatted_tweet.extend([np.nan, np.nan])
        try:
            formatted_tweet.extend([[media['media_url'] for media in tweet['extended_entities']['media']],
                                    [media['type'] for media in tweet['extended_entities']['media']]])
        except Exception:
            formatted_tweet.extend([np.nan, np.nan])
        formatted_tweet.extend([tweet['in_reply_to_status_id'], tweet['in_reply_to_user_id'],
                                tweet['in_reply_to_screen_name'], tweet['user']['id'],
                                tweet['user']['name'], tweet['user']['screen_name'],
                                tweet['user']['location'], tweet['user']['description'],
                                tweet['user']['url'], tweet['user']['followers_count'],
                                tweet['user']['friends_count'], tweet['user']['listed_count'],
                                tweet['user']['created_at'], tweet['user']['favourites_count'],
                                tweet['user']['verified'], tweet['user']['statuses_count'], tweet['user']['lang'],
                                tweet['user']['contributors_enabled'], tweet['contributors'],
                                tweet['is_quote_status'], tweet['retweet_count'], tweet['favorite_count'],
                                tweet['favorited'], tweet['retweeted'], tweet['lang']])
        try:
            formatted_tweet.append(tweet['possibly_sensitive'])
        except Exception:
            formatted_tweet.append(np.nan)
        formatted_tweet.append(query)
        formatted_tweet.append(orientation)
        formatted_tweet.append(datetime.datetime.now().strftime("%d-%m-%Y"))
        formatted_tweets.append(formatted_tweet)
    print('{} tweets formatted'.format(len(formatted_tweets)))
    return formatted_tweets


def tweets_to_df(formatted_tweets):
    """
    Turns the list of formatted tweets to a dataframe
    :param formatted_tweets: list of tweets (list format) with only the relevant fields
    :return: dataframe of formatted tweets
    """
    col = ['created_at', 'id_number', 'text', 'truncated', 'text_range', 'hashtags', 'mentions_screen_names',
           'mentions_names', 'links', 'media_urls', 'media_types', 'expanded_media_urls', 'expanded_media_types',
           'reply_to_status_id', 'reply_to_user_id', 'reply_to_screen_name', 'user_id', 'user_name', 'user_screen_name',
           'user_location', 'user_description', 'user_url', 'user_followers_count', 'user_friends_count',
           'user_listed_count', 'user_created_at', 'user_favourites_count', 'user_verified', 'user_statuses_count',
           'user_lang', 'user_contributors_enabled', 'contributors', 'is_quote_status', 'retweet_count',
           'favorites_count', 'favorited', 'retweeted', 'lang', 'possibly_sensitive', 'query', 'orientation', 'date']
    tweets_df = pd.DataFrame(formatted_tweets, columns=col)
    return tweets_df


def save_df(tweets_df, path):
    """
    Saves the dataframe to csv
    :param tweets_df: Dataframe of formatted tweets
    :param path: where to save the tweets
    :return: None
    """
    tweets_df.dropna(subset=['text', 'id_number'])
    tweets_df.to_csv(path)
    print('Dataframe saved to file')


def download_and_save_tweets():
    """
    General function that downloads, formats, and saves the tweets
    :param query: hashtag to query
    :param limit: max number of tweets to download
    :param path: where to save the csv
    :return: formatted data frame of tweets
    """
    path = input('Please enter the path of the project directory: ')
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    queries = [x for x in input('Hashtags to query (separated by " ,"): ').split(', ')]
    orientations = [x for x in input('Orientation of the hashtags (pro/against, separated by " ,"): ').split(', ')]
    limit = input('Maximum number of tweets to download: ')
    # Moving to the right directory and checking if a folder for raw data exists (creates it if not)
    os.chdir(path)
    if not os.path.exists('./raw_data'):
        os.makedirs('./raw_data')
    os.chdir('./raw_data')
    # Downloading and saving the data
    for query, orientation in zip(queries, orientations):
        tweets = download_tweets(query, limit)
        formatted_tweets = format_tweets(tweets, query, orientation)
        tweets_df = tweets_to_df(formatted_tweets)
        save_df(tweets_df, query + '_' + date + '.csv')
    print('All hashtags downloaded')
    return None


if __name__ == "__main__":
    download_and_save_tweets()
