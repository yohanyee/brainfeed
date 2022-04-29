#!/usr/bin/env python

# %% [markdown]
# # Search Reddit Posts and Do Some Analyses

# %%
# Import custom function
from reddit_pushshift import search_reddit

# %%
# Import other packages
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Search The Most Recent Reddit Posts in the Science Subreddit

# %%
# Search reddit posts and store the results in a dataframe

# Arguments
# subreddit='science'
# n_posts = 100
# start_epoch = int(dt.datetime(2020, 1, 1).timestamp()) # starting date
# end_epoch = int(dt.datetime(2020, 12, 31).timestamp()) # ending date
# keyword

# If a time window is not defined, it searches the most recent posts
start_epoch = int(dt.datetime(2021, 1, 1).timestamp()) # starting date
end_epoch = int(dt.datetime(2021, 12, 31).timestamp()) # ending date
posts = search_reddit(
    subreddit='science',
    n_posts=1000
)
posts.info()

# %% [markdown]
# ## Overview of the Posts Found

# %%
posts.head()

# %%
# Visualize the number of posts for each flair tag
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.countplot(data=posts, x='link_flair_text')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_xlabel("Flair Tag")

sns.despine()
plt.show()

# %% [markdown]
# ## Simple Binary Classification 
# Predict whether a post's flair tag is "Psychology" or "Neuroscience" based on the post content ("title")

# %%
# Extract posts with flair tags of Psychology and Neuroscience
sel_tags = ['Psychology', 'Neuroscience']
idx = posts['link_flair_text'].str.contains('Psychology|Neuroscience', regex=True)
posts_sel = posts.loc[idx,:]
posts_sel = posts_sel.drop_duplicates(subset=['url']) # drop duplicated posts
posts_sel.info()

# %%
posts_sel.head()

# %%


# %%
