#!usr/bin/python

# Use the Python Pushshift.io API wrapper to search Reddit posts and comments

# In addition to the praw package, you also need to install psaw
# pip install psaw

# Import packages
import praw
from psaw import PushshiftAPI
import pandas as pd
import datetime as dt

def search_reddit(subreddit='science', n_posts=100, start_epoch=None, end_epoch=None, keyword=None):
    # # Initialize the Reddit API via PRAW
    # # make sure you have properly defined necessary information in praw.ini file (in the $HOME/.config directory)
    # reddit = praw.Reddit("brainfeed_script")
    # api = PushshiftAPI(reddit)

    # Somehow the above doesn't work, try directly initialize the Pushshift API
    api = PushshiftAPI()
    
    # attributes we are looking for
    attrb = ['author','id','title','selftext','score','url',
             'num_comments','subreddit','created_utc','link_flair_text','link_flair_type']
    
    # Search
    # start_epoch = int(dt.datetime(2020, 1, 1).timestamp()) # starting date
    # end_epoch = int(dt.datetime(2020, 12, 31).timestamp()) # ending date
    gen = api.search_submissions(
        filter=attrb,
        subreddit=subreddit, # search in a given subreddit
        limit=n_posts, # the number of submissions returned 
        after=start_epoch, # can only define the starting or ending date, or define both
        before=end_epoch,
        q=keyword # key term to search for
    )
    posts = list(gen)
    
    # Store the results into a dataframe
    df_posts = pd.DataFrame(columns=attrb)
    for i,post in enumerate(posts):
        if hasattr(post, 'link_flair_text'):
        # some submissions don't have the 'link_flair_text' attribute
            df_posts.loc[i,:] = [
                post.author, post.id, post.title, post.selftext,
                post.score, post.url, post.num_comments,post.subreddit,
                post.created_utc, post.link_flair_text, post.link_flair_type
            ]
        else:
            df_posts.loc[i,:] = [
                post.author, post.id, post.title, post.selftext,
                post.score, post.url, post.num_comments,post.subreddit,
                post.created_utc, None, post.link_flair_type
            ]
    # convert the UTC timestamp to readable date
    df_posts["created_time"] = pd.to_datetime(df_posts['created_utc'], unit='s') 
    # df_posts.head()
    
    return df_posts
