
# Input: screen name of account
# Output: up to 200 most recent retweets of the account
from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import json
import pickle
import codecs
from tweepy.utils import parse_datetime

# Step 1: Authentication
# Go to http://apps.twitter.com and create an app.
# Copy and paste the keys and tokens below
consumer_key="f5vfC0cTaHA6AKLOwQHtdImOX"
consumer_secret="a3skaiwaheh7jbjbgOJyu3yMRVnqJQPxsH7OfZi7ZAZVMEoNuT"
access_token="971917970829185024-T6gI59cPkBeNfGWYWAJynIvlzUsA2Lc"
access_token_secret="5Z1juX40eKwt9gtVLDfdcvXesKSW8O4zCbOmJun0JBYoh"

# Step 2: Provide Input 
# Provide the screen_name (id) of the Twitter Account you want to scrape
screen_name = 'realDonaldTrump'
# ****** Ready to Run *********


auth = OAuthHandler(consumer_key, consumer_secret)
auth.secure = True
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, parser=tweepy.parsers.JSONParser(), retry_count=5)

rate = api.rate_limit_status()
rate_follower_IDs =  rate['resources']['followers']['/followers/ids']['limit'] * 4 # requests per hour
rate_follower_profiles =  rate['resources']['users']['/users/lookup']['limit'] * 4 # requests per hour

# doc explaning each field https://dev.twitter.com/overview/api/tweets
tweet_cols = ['id_str', 'user.screen_name', 'created_at', 'retweeted_status.id_str',
             'retweeted_status.user.id_str', 'retweeted_status.user.screen_name',
             'retweeted_status.favorite_count', 'favorite_count', 'retweet_count', 'text']

## Print a list of dict variable into csv file
# vals: a list, each element is a dict
# cols: the fields (keys) to be retrieved in each dict
def printList(vals, cols, file_name):
    f = codecs.open(file_name, 'w', 'utf-8')
    f.writelines('\t'.join(cols)+'\n')
    for v in  vals:
        if v==None: continue
        fields = [getDictField(v, e) for  e in  cols]
        f.writelines('\t'.join(fields)+'\n')
    f.close()       

## Access an element in a nested dict variable by key
# d: a dict
# field: the field to be retrieved. seperated by '.' if nested
def getDictField(d, field):
    names = field.split('.')
    for name in  names:
        if name in d: d = d[name]
    try:
        if "created_at" in field: d = tweepy.utils.parse_datetime(d)
        if 'text' in field: d = d.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        return str(d)
    except:
        print("@@@Exception while extracting the field %s from the following object" % field)
        print(d)
        return "fail_to_convert_to_string"



if __name__ == '__main__':
    # Scrape the most recent N (up to 200) tweets authored by the given id
    tweets = api.user_timeline(screen_name=screen_name, count=200)
    printList(tweets, tweet_cols, screen_name+'.tsv')
    print('Done!')


