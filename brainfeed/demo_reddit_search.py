#!/usr/bin/env python

# Demonstrate how to use the search_reddit function
# This script requests user inputs and returns a pandas dataframe storing the searched reddit posts

# Run this demo by entering the following command in the Terminal: python demo_reddit_search.py 

# Import custom function
from app_reddit_search.reddit_pushshift import search_reddit

# Import other packages
# import pandas as pd
import datetime as dt
from pathlib import Path

# Request user input
# subreddit
subreddit = input("Subreddit (leave it blank to search posts from all subreddits): ")
if not subreddit: # no user input (empty string)
    subreddit = None
# number of posts to be returned
while True:
    try:
        n_posts = input("Number of posts to be returned (enter an integer > 0; Default = 50): ")
        if not n_posts: # no user input (empty string)
            n_posts = 50
        elif int(n_posts) <= 0:
            raise ValueError
        else:
            n_posts = int(n_posts)
        break
    except ValueError: 
        print("Invalid input. Please try again!")
# start date
while True:
    try:
        start_date = input("Earliest posts to be searched (yyyy-mm-dd; leave it blank to search all posts): ")
        if not start_date: # no user input (empty string)
            start_date = None
        else:
            year, month, date = start_date.split('-')
            year = int(year)
            month = int(month)
            date = int(date)
            start_date = int(dt.datetime(year, month, date).timestamp())
            del year, month, date
        break
    except ValueError:
        print("Invalid input. Please try again!")
# end date
while True:
    try:
        end_date = input("Latest posts to be searched (yyyy-mm-dd; leave it blank to search all posts): ")
        if not end_date: # no user input (empty string)
            end_date = None
        else:
            year, month, date = end_date.split('-')
            year = int(year)
            month = int(month)
            date = int(date)
            end_date = int(dt.datetime(year, month, date).timestamp())
            del year, month, date
        break
    except ValueError:
        print("Invalid input. Please try again!")    
# keyword
keyword = input("Keyword to be searched (currently only 1 keyword is allowed; leave it blank to search all posts without targeting a specific keyword): ")
if not keyword: # no user input (empty string)
    keyword = None

# Search posts
print("Searching...")
posts = search_reddit(
    subreddit=subreddit,
    n_posts=n_posts,
    start_date=start_date,
    end_date=end_date,
    keyword=keyword
)
print("Search is done!")

# Display the first 5 results (not every column is shown)
# print(posts[['author','id','title','url','subreddit','link_flair_text','created_time']].head())
if posts.shape[0]==0:
    print("No Posts Found")
print(posts[['id','author','title','url','subreddit','link_flair_text','created_time']].head().to_markdown())

# Prompt to ask whether save the results
while True:
    try:
        savePost = input("Save the search posts/results ([y]/n)? ")
        if not savePost: # no input (empty string)
            savePost = "y"
        assert savePost=="y" or savePost=="n"
        if savePost=="y":
            print("Saving the search results")
            # define the output directory
            cwd = Path.cwd()
            # rootDir = cwd.parent.absolute()
            outDir = cwd / 'results_reddit_search' # outDir = rootDir / 'results_reddit_search'
            if not outDir.is_dir(): # create the directory if not exist
                outDir.mkdir()
            # save
            fileName = outDir / ("reddit_search_" + dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".csv")
            posts.to_csv(fileName, index=False)
        elif savePost=="n":
            print("NOT saving the search results")
        break
    except AssertionError:
        print("Invalid input")
    