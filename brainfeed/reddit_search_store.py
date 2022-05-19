#!/usr/bin/env python

# %% [markdown]
# # Search Reddit Posts and Store into a Database
# For demonstration purpose, this script only searches the most recent posts from the 'science' subreddit

# %%
# Import custom function
from app_reddit_search.reddit_pushshift import search_reddit

# %%
# Import other packages
# import pandas as pd
# import numpy as np
# import datetime as dt
from pathlib import Path
# import os

import psycopg2

# import matplotlib.pyplot as plt
# import seaborn as sns

# %% [markdown]
# ## Search The Most Recent Reddit Posts in the Science Subreddit

# %%
# Search reddit posts and store the results in a dataframe

# Arguments
# subreddit='science'
# n_posts = 100
# start_date = int(dt.datetime(2020, 1, 1).timestamp()) # starting date
# end_date = int(dt.datetime(2020, 12, 31).timestamp()) # ending date
# keyword

# If a time window is not defined, it searches the most recent posts
# start_date = int(dt.datetime(2021, 1, 1).timestamp()) # starting date
# end_date = int(dt.datetime(2021, 12, 31).timestamp()) # ending date
# keyword = None
posts = search_reddit(
    subreddit='science',
    n_posts=5000
)
posts.info()

# %% [markdown]
# ## Store The Search Results into a PostgreSQL DataBase

# %%
# Establish the connection
conn = None
try:
    conn = psycopg2.connect(
        user='postgres', password='', host='localhost', port= '5432'
    ) # database='postgres', 
    print("Postgres server is connected")
except psycopg2.OperationalError:
    print("Postgres server is not running.")
    raise

# Create a new database reddit (if not created yet)
if conn is not None:
    conn.autocommit = True
    cursor = conn.cursor() # Create a cursor object using the cursor() method
    cursor.execute("SELECT datname FROM pg_database;")
    list_database = cursor.fetchall()
    
    dbname = 'reddit'
    if (dbname,) in list_database:
        print("'{}' Database exists".format(dbname))
    else:
        print("'{}' Databased does not exist".format(dbname))
        print("Create it...")
        sql_createDB = "CREATE database " + dbname + ";"
        cursor.execute(sql_createDB) # create a database
        print("'{}' Database is now created.".format(dbname))

# Close the connection
conn.close()
cursor.close()

# %%
# Connect to the reddit database
try:
    conn = psycopg2.connect(
        database='reddit', user='postgres', password='', host='localhost', port= '5432'
    )
    print("Connect to the reddit database")
    conn.autocommit = True
    cursor = conn.cursor() # Create a cursor object using the cursor() method
except psycopg2.OperationalError:
    print("Postgres server is not running or the reddit database does not exist.")
    raise
    
# Show existing tables
sql_show_table = '''
SELECT table_schema, table_name
FROM information_schema.tables
WHERE (table_schema = 'public');
'''
cursor.execute(sql_show_table)
list_tables = cursor.fetchall()
# print(list_tables)
if ('public', 'reddit_search') in list_tables:
    print("reddit_search TABLE exists.")
else:
    print("reddit_search TABLE does not exist.")

# Create a table to store all search results
try:
    tbl_search_create_sql = '''
    CREATE TABLE IF NOT EXISTS reddit_search (
        id VARCHAR PRIMARY KEY,                
        author VARCHAR,
        title VARCHAR,
        selftext VARCHAR,
        score INTEGER,
        url VARCHAR,
        num_comments INTEGER,
        subreddit VARCHAR,
        created_utc INTEGER,
        link_flair_text VARCHAR,
        link_flair_type VARCHAR,
        created_time TIMESTAMP,
        updated_time TIMESTAMP DEFAULT current_timestamp
    );
    '''
    cursor.execute(tbl_search_create_sql)
    conn.commit()
except SyntaxError:
    print("Error when creating the main table reddit_search")
    raise

# Create a table to store the current search results
# The current search results will be later inserted into the main table that stores all results (exclude duplicate)
try:
    tbl_tmp_create_sql = '''
    CREATE TABLE IF NOT EXISTS tmp_reddit_search (
        id VARCHAR PRIMARY KEY,                
        author VARCHAR,
        title VARCHAR,
        selftext VARCHAR,
        score INTEGER,
        url VARCHAR,
        num_comments INTEGER,
        subreddit VARCHAR,
        created_utc INTEGER,
        link_flair_text VARCHAR,
        link_flair_type VARCHAR,
        created_time TIMESTAMP,
        updated_time TIMESTAMP DEFAULT current_timestamp
    );
    '''
    cursor.execute(tbl_tmp_create_sql)
    conn.commit()
except SyntaxError:
    print("Error when creating the temporary table tmp_reddit_search")
    raise

# # Create a table to store author name (and create unique ID)
# tbl_author_create_sql = '''
# CREATE TABLE IF NOT EXISTS author (
#     author_id serial PRIMARY KEY
#     author_name VARCHAR UNIQUE NOT NULL
# );
# '''
# cursor.execute(tbl_author_create_sql)
# conn.commit()

# %%
# Insert the dataframe "posts" containing the search results into the table search_reddit
# Use the SQL COPY method

# (0) Check whether the connection is alive and if it's connected to the correct database
assert conn.closed == 0, "The connection is closed. Reconnect it!"
assert conn.info.dbname == 'reddit', "Not connect to the reddit database"
# (0) Check there is no duplicate post id
assert len(posts['id'].unique()) == posts.shape[0], "There is at least 1 duplidate post id. Check!"

# (1) Save the dataframe to a csv (temporary)
cwd = Path.cwd()
# rootDir = cwd.parent.absolute()
# outDir = rootDir / 'results_reddit_search'
fileName = "tmp_reddit_search.csv" #outDir / ("tmp_reddit_search.csv")
posts.to_csv(fileName, index=False, header=False)

# (2) Clear data in the tmp_reddit_search TABLE
try:
    cursor.execute('TRUNCATE TABLE tmp_reddit_search') 
    conn.commit()
except SyntaxError:
    print("Error when clear data in TABLE tmp_reddit_search") 
    raise

# (3) COPY csv to the TABLE tmp_reddit_search
# Note: COPY FROM Copies data from a file to a table (appending the data to whatever is in the table already)
cols_str = "(id, author, title, selftext, score, url, num_comments, subreddit, created_utc, link_flair_text, link_flair_type, created_time)"
sql_copy = "COPY tmp_reddit_search " + cols_str + " FROM '" + str(cwd/fileName) + "' DELIMITER ',' CSV;"
try:
    cursor.execute(sql_copy)
    conn.commit()
except SyntaxError:
    print("Error when COPY the current search results into TABLE tmp_reddit_search")
    raise

# (4) Insert the current search results to the main TABLE reddit_search
# Note: if a post already exists in the main table, did not insert the duplicate post (post id) returned in the current search results
cols_str_nopara = "id, author, title, selftext, score, url, num_comments, subreddit, created_utc, link_flair_text, link_flair_type, created_time"
sql_upsert = "INSERT INTO reddit_search " + cols_str + " SELECT " + cols_str_nopara + " FROM tmp_reddit_search ON CONFLICT (id) DO NOTHING;" 
try:
    cursor.execute(sql_upsert)
    conn.commit()
except SyntaxError:
    print("Error when insert the current search results into the TABLE reddit_search")
    raise

# Close the connection
conn.close()
cursor.close()

# %%