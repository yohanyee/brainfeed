#!/usr/bin/env python

# Use the Python Pushshift.io API wrapper to search Reddit posts and comments

# Make sure the praw and psaw packages are installed
# pip install praw
# pip install psaw

# Import packages
import praw
from psaw import PushshiftAPI
import pandas as pd
# import datetime as dt

# This function use the Python Pushshift.io API wrapper to search Reddit posts and comments
def search_reddit(subreddit=None, n_posts=50, start_date=None, end_date=None, keyword=None):
    """Search Reddit Posts

    Args:
        subreddit (str, optional): a string indicating the subreddit to be searched. Defaults to None (i.e., search from all subreddits).
        n_posts (int, optional): an integer indicating the number of posts to be returned in the search results. Defaults to 50.
        start_date (int, optional): an integer of UTC timestamp indicating the earliest post to be searched. Defaults to None.
        end_date (int, optional): an integer of UTC timestamp indicating the latest post to be searched. Defaults to None.
        keyword (str, optional): a string indicating the keyword to be searched (i.e., posts contain the keyword). Defaults to None.

    Returns:
        pd.DataFrame: a pandas dataframe storing the searched results
    """
    
    # # Initialize the Reddit API via PRAW
    # # make sure you have properly defined necessary information in praw.ini file (in the ~/.config directory (Mac))
    # reddit = praw.Reddit("brainfeed_script")
    # api = PushshiftAPI(reddit)

    # Somehow the above doesn't work, try directly initialize the Pushshift API
    api = PushshiftAPI()
    
    # attributes we are looking for
    attrb = ['id','author','title','selftext','score','url',
             'num_comments','subreddit','created_utc','link_flair_text','link_flair_type']
    # https://praw.readthedocs.io/en/latest/code_overview/models/submission.html
    # author: Redditor (the poster)
    # id: id of the submission
    # score: the number of upvotes for the submission
    # selftext: the submissions’ selftext - an empty string if a link post
    
    # Search
    # start_date = int(dt.datetime(2020, 1, 1).timestamp()) # starting date
    # end_date = int(dt.datetime(2020, 12, 31).timestamp()) # ending date
    gen = api.search_submissions(
        filter=attrb,
        subreddit=subreddit, # search in a given subreddit
        limit=n_posts, # the number of submissions returned 
        after=start_date, # can only define the starting or ending date, or define both
        before=end_date,
        q=keyword # key term to search for
    )
    posts = list(gen)
    
    # Store the results into a dataframe
    df_posts = pd.DataFrame(columns=attrb)
    for i,post in enumerate(posts):
        if hasattr(post, 'link_flair_text'):
        # some submissions don't have the 'link_flair_text' attribute
            df_posts.loc[i,:] = [
                post.id, post.author, post.title, post.selftext,
                post.score, post.url, post.num_comments,post.subreddit,
                post.created_utc, post.link_flair_text, post.link_flair_type
            ]
        else:
            df_posts.loc[i,:] = [
                post.id, post.author, post.title, post.selftext,
                post.score, post.url, post.num_comments,post.subreddit,
                post.created_utc, None, post.link_flair_type
            ]
    # convert the UTC timestamp to readable date
    df_posts["created_time"] = pd.to_datetime(df_posts['created_utc'], unit='s') 
    # df_posts.head()
    
    # Define datatype
    df_posts = df_posts.astype({
        'score': 'int64',
        'num_comments': 'int64',
        'created_utc': 'int64'                
    })
    
    return df_posts
