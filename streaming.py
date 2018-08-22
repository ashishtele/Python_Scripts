
# Input: keywords
# Output: tweets related to those keywords
from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import json
import pickle
import codecs

# Step 1: Authentication
# Go to http://apps.twitter.com and create an app.
# Copy and paste the keys and tokens below
consumer_key="f5vfC0cTaHA6AKLOwQHtdImOX"
consumer_secret="a3skaiwaheh7jbjbgOJyu3yMRVnqJQPxsH7OfZi7ZAZVMEoNuT"
access_token="971917970829185024-T6gI59cPkBeNfGWYWAJynIvlzUsA2Lc"
access_token_secret="5Z1juX40eKwt9gtVLDfdcvXesKSW8O4zCbOmJun0JBYoh"

# Step 2: Provide Input 
# Which keywords to collect data on and how many tweets to collect
keywords = ['Apple','Google','Uber']
num_tweets = 200
# ****** Ready to Run *********


tweets =  [] # store collected tweets
tsv_file = "stream_tweets_%s.tsv" % '_'.join(keywords)
# Fields to be extracted from returned data
tweet_cols = ['id_str', 'user.id_str', 'user.screen_name', 'created_at', 'retweeted_status.id_str',
              'retweeted_status.user.id_str', 'retweeted_status.user.screen_name',
              'retweeted_status.favorite_count', 'favorite_count', 'retweet_count',  'text']

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

## A class that defines a set of actions while streaming data
class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    # what to do with returned data, once a tweet (real-time)
    def on_data(self, raw_data):
        try:
            data =  json.loads(raw_data)
        except:
            print('@@@ Exception: unable to parse the following data, will skip')
            print(raw_data)
            return True
        if 'retweeted_status' in data or data['in_reply_to_status_id_str']: return True
        if data and len(data)>0: tweets.append(data) 
        if len(tweets) >= num_tweets:
            print('Stopped as %d tweets had been collected' % len(tweets))
            return False  # streaming will stop if return False
        print('%d tweets collected' % len(tweets))
        printList(tweets, tweet_cols, tsv_file)
        return True

    # what to do on error
    def on_error(self, status):
        print(status)
        # return False  # if uncommented, will stop on error

if __name__ == '__main__':
    # authentication
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # initialize a Stream object to call streaming API
    stream = Stream(auth, StdOutListener())
    stream.filter(track=keywords, languages=['en'])
    
    # save results as csv files
    printList(tweets, tweet_cols, tsv_file)
    print('****Done!******')
