import tweepy
import configparser
import sys
import pandas as pd
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

def pull_tweets(hashtag):
    tweets = tweepy.Cursor(api.search, q=hashtag, lang="en").items(2000)
    return tweets

def clean(tweet):
    words = word_tokenize(tweet)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if not word in stop_words]
    pos_tags = nltk.pos_tag(words) 

    lemmatizer = WordNetLemmatizer()
    for i in range(len(words):
        words[i] = lemmatizer.lemmatize(words[i], pos=pos_tags[i][1])

    return words

def create_df(tweets):
    df = pd.DataFrame()

    df['id'] = [tweet.id for tweet in tweets]
    df['date'] = [tweet.created_at for tweet in tweets]
    df['text'] = [tweet.text for tweet in tweets]
    df['all_hashtags'] = [tweet.entities.get('hashtags') for tweet in tweets]
    df['cleaned_text'] = [clean(tweet.text) for tweet in tweets]
    return df

def get_polarity(tweet):
    score = analyzer.polarity_scores(" ".join(tweet)
    return tweet

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    auth = tweepy.OAuthHandler(config['CONSUMER_KEY'], config['CONSUMER_SECRET'])
    auth.set_access_token(config['ACCESS_TOKEN'], config['ACCESS_TOKEN_SECRET'])

    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = pull_tweets(f"#{sys.argv[1]}")

    data = create_df(tweets)

    for i in range(0, data.shape(0)):
        df.loc[i, 'polarity'] = get_polarity(df.loc[i]['cleaned_text'])

    data.to_excel(f"{sys.argv[1]}.xlsx")
    print(f"Data written to {sys.argv[1]}.xlsx")


if __name__ == "__main__":
    main()
