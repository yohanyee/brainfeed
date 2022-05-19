#!/usr/bin/env python

from flask import Flask, render_template, request, redirect, url_for, session
from app_reddit_search import app
from flask_session import Session

from app_reddit_search.reddit_search_form import SearchForm
from app_reddit_search.reddit_pushshift import search_reddit

import pandas as pd
import datetime as dt

Session(app)

@app.route("/")
@app.route("/search", methods=['GET', 'POST'])
def search():
    form = SearchForm(request.form)
    if session.get('posts'): 
        # clear previous search results if they exist
        session['posts'] = None
    if request.method == "POST" and form.validate_on_submit():
        # subreddit
        subreddit = request.form.get('subreddit')
        if not subreddit: # no user input (empty string)
            subreddit = None
        # number of posts to be returned    
        n_posts = request.form.get('n_posts')
        if not n_posts: # no user input (empty string)
            n_posts = 5
        else:
            n_posts = int(n_posts)
        # start date
        start_date = request.form.get('start_date')
        if not start_date: # no user input (empty string)
            start_date = None
        else:
            year, month, date = start_date.split('-')
            year = int(year)
            month = int(month)
            date = int(date)
            start_date = int(dt.datetime(year, month, date).timestamp())
            del year, month, date
        # end date
        end_date = request.form.get('end_date')
        if not end_date: # no user input (empty string)
            end_date = None
        else:
            year, month, date = end_date.split('-')
            year = int(year)
            month = int(month)
            date = int(date)
            end_date = int(dt.datetime(year, month, date).timestamp())
            del year, month, date
        # keyword
        keyword = request.form.get('keyword')
        if not keyword: # no user input (empty string)
            keyword = None
        
        posts = search_reddit(
            subreddit=subreddit,
            n_posts=n_posts,
            start_date=start_date,
            end_date=end_date,
            keyword=keyword
        )
        cols_sel = ['author','id','title','url','subreddit','link_flair_text','created_time']
        posts = posts[cols_sel]
        session['posts'] = posts.to_dict('list')
        return redirect(url_for('results')) #**form.data
        
        # data = [posts.to_html(classes='data',header=True,index=False)]
        # return render_template("search.html", title="Search", form=form, data=data)
        
    return render_template("search.html", title="Search", form=form)

@app.route("/results", methods=['GET', 'POST'])
def results():    
    posts_dict = session.get('posts')
    posts = pd.DataFrame(posts_dict)
    # print(posts)
    return render_template(
        "results.html", title="Results",
        data=[posts.to_html(classes='mystyle',header=True,index=True)]
        ) 
    # headings=posts.columns.values,
    # classes='table table-stripped'
