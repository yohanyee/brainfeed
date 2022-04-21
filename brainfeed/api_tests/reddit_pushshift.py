#!usr/bin/python

# Use the Python Pushshift.io API wrapper to search Reddit posts and comments

# In addition to the praw package, you also need to install psaw
# pip install psaw

# Import packages
import praw
from psaw import PushshiftAPI
import pandas as pd
import datetime as dt

# # Initialize the Reddit API via PRAW
# # make sure you have properly defined necessary information in praw.ini file (in the $HOME/.config directory)
# reddit = praw.Reddit("brainfeed_script")
# api = PushshiftAPI(reddit)

# Somehow the above doesn't work, try directly initialize the Pushshift API
api = PushshiftAPI()

# Try to search for the most recent submissions/posts in the 'science' subreddit
# the number of submissions returned is defined by the argument "limit"
# (1) Search 
# attributes we are looking for
attrb = ['author','id','title','selftext','score','url','num_comments','subreddit','created_utc','link_flair_text','link_flair_type']
gen = api.search_submissions(
    subreddit='science',
    filter=attrb,
    limit=100000
)
posts = list(gen)
# print(posts)
# (2) Store the restuls into a dataframe
df_posts = pd.DataFrame(columns=attrb)
for i,post in enumerate(posts):
    df_posts.loc[i,:] = [
        post.author, post.id, post.title, post.selftext,
        post.score, post.url, post.num_comments,post.subreddit,
        post.created_utc, post.link_flair_text, post.link_flair_type
    ]
df_posts["created_time"] = pd.to_datetime(df_posts['created_utc'], unit='s') # convert the UTC timestamp to readable date
df_posts.head()

# Try to search for submissions in a given time period (e.g., from 2020/1/1 to present) in the 'science' subreddit
# the number of submissions returned is defined by the argument "limit"
# (1) Define the time period
# you can only define the starting or ending date, or define both
start_epoch=int(dt.datetime(2020, 1, 1).timestamp()) # starting date
end_epoch=int(dt.datetime(2020, 12, 31).timestamp()) # ending date
# (2) Search
gen = api.search_submissions(
    after=start_epoch,
    before=end_epoch,
    subreddit='science',
    filter=attrb,
    limit=10
) 
posts_twin = list(gen) # posts in a given time window
# (3) Store the restuls into a dataframe
df_posts_twin = pd.DataFrame(columns=attrb)
for i,post in enumerate(posts_twin):
    df_posts_twin.loc[i,:] = [
        post.author, post.id, post.title, post.selftext,
        post.score, post.url, post.num_comments,post.subreddit,
        post.created_utc, post.link_flair_text, post.link_flair_type
    ]
df_posts_twin["created_time"] = pd.to_datetime(df_posts_twin['created_utc'], unit='s') # convert the UTC timestamp to readable date
df_posts_twin.head()

