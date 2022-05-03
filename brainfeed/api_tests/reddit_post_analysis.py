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
# start_date = int(dt.datetime(2020, 1, 1).timestamp()) # starting date
# end_date = int(dt.datetime(2020, 12, 31).timestamp()) # ending date
# keyword

# If a time window is not defined, it searches the most recent posts
start_date = int(dt.datetime(2021, 1, 1).timestamp()) # starting date
end_date = int(dt.datetime(2021, 12, 31).timestamp()) # ending date
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
ax.set_title("Number of Posts for Each Flair Tag")

sns.despine()
plt.show()

# %% [markdown]
# ## Simple Binary Classification 
# Predict whether a post's flair tag is "Psychology" or "Neuroscience" based on the post content ("title")

# %%
# Extract posts with flair tags of Psychology and Neuroscience
sel_tags = ['Environment','Biology']#['Psychology', 'Neuroscience']
search_tags = 'Environment|Biology'#'Psychology|Neuroscience'
idx = posts['link_flair_text'].str.contains(search_tags, regex=True)
posts_sel = posts.loc[idx,:]
posts_sel = posts_sel.drop_duplicates(subset=['url']) # drop duplicated posts
posts_sel.info()

# %%
posts_sel.head()

# %%
post_titles = posts_sel['title']
post_tags = posts_sel['link_flair_text']

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# %%
# Split data into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    post_titles, post_tags, test_size=0.3,
    stratify=post_tags, random_state=0
)

# Number of samples
print("Number of total training samples:", X_train.shape[0])
print("Number of total testing samples:", X_test.shape[0])

# Number of samples in each class
print(
    "Number of training samples with tag {} and {}, respectivly: {} and {}".format(
        sel_tags[0], sel_tags[1],
        y_train.value_counts().loc[sel_tags[0]],
        y_train.value_counts().loc[sel_tags[1]]
    )
)
print(
    "Number of testing samples with tag {} and {}, respectivly: {} and {}".format(
        sel_tags[0], sel_tags[1],
        y_test.value_counts().loc[sel_tags[0]],
        y_test.value_counts().loc[sel_tags[1]]
    )
)

# %% [markdown]
# ### CountVectorizer

# %%
# Count Vectorizor
count_vectorizer = CountVectorizer(stop_words='english') #strip_accents='unicode', 
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)

print(count_train.shape)
print(count_test.shape)

# %%
count_nb_clf = MultinomialNB()
count_nb_clf.fit(count_train, y_train.ravel())
count_pred = count_nb_clf.predict(count_test)
print("Naive Bayes Classification Accuracy Based on Word Count: {:.2f}".format(
    accuracy_score(y_test, count_pred)
    )
)
print(
    classification_report(y_test, count_pred)
)

# %% [markdown]
# ### TF-IDF Vectorizer

# %%
tfidf_vectorizer = TfidfVectorizer(stop_words='english') #strip_accents='unicode',
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
tfidf_test = tfidf_vectorizer.transform(X_test.values)

# %%
tfidf_nb_clf = MultinomialNB()
tfidf_nb_clf.fit(tfidf_train, y_train.ravel())
tfidf_pred = tfidf_nb_clf.predict(tfidf_test)
print("Naive Bayes Classification Accuracy Based on TF-IDF Matrix: {:.2f}".format(
    accuracy_score(y_test, tfidf_pred)
    )
)
print(
    classification_report(y_test, tfidf_pred)
)

# %%
