# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:19:51 2020

@author: Andrew

This is the actual NLP part of the project.

This script is going to:
    1. Check if the data in the reddit_parse file has been updated recently (3 days)
        If not updated - update the file
    2. Sentiment analysis based on only the title of the post
        What do people think is good, what do people think is bad on a high level.
        This is will be done using VADER and can be done on every single post.
    3. Topic modeling of larger text posts
        This will be done like the TFIDF work I did for other NLP projects
    4. Sentiment analysis and topic modeling for comments on posts.
        For longer comments perhaps TFIDF is going to be better than VADER for finding sentiment
"""
from reddit_extract import subreddit
import pandas as pd
import os
import time

wdsr = subreddit('watch_dogs')

# function that checks when the file was last updated
# if it was updated more than 3 days ago run the save_posts process.
# should return the dataframe that is the reddit file
def get_data(subreddit,how='hot'):
    path_to_file = '{}-{}.csv'.format(subreddit.subreddit,how)
    
    # try and check when the file was last modified
    # if the path doesnt exist then make the file
    try:
        mod_time = os.path.getmtime(path_to_file)
        cur_time = time.time()
        
        if (cur_time-mod_time) > 43200:
            print('File out of date - updating')
            subreddit.save_posts()
        
        df = pd.read_csv(path_to_file)
        
    except:
        subreddit.save_posts()
        df = pd.read_csv(path_to_file)
    
    print('Data loaded')
    return df

def scatter_plot(x,y,title=None,xlabel=None,ylabel=None,size=50,
                 save=False,add_avg=False,add_med=False
                 ):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12,9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=16)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)
    
    if title is not None:
        plt.title(title,fontsize=18)
    
    plt.scatter(x=x,y=y,s=size)
    
    if add_avg:
        plt.plot([np.mean(y)]*len(x),'r--')
    
    if add_med:
        plt.plot([np.median(y)]*len(x),'g--')
        
    if save:
        if title is None:
            title='figure01'
        else:
            title=title.replace(' ','_')
            
        plt.savefig('{}.png'.format(title))
        print('Saved plot to {}.png'.format(title))
    else:
        plt.show()
    

df = get_data(wdsr)

#### STEP 1 ####
    
################
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

# take the columns that are either positive or negative. a compound score of 0 is not interesting
for t in df['title']:
    score = sia.polarity_scores(t)
    score['title'] = t
    if score['compound'] != 0:
        results.append(score)

vader_df = pd.DataFrame(results)

xplot = range(len(vader_df['compound']))
yplot = vader_df['compound']

scatter_plot(x=xplot,
             y=yplot,
             xlabel='Post Number',
             ylabel='Compound VADER Score',
             title='VADER Scores for Watch_Dogs Submission Titles',
             add_avg=True,
             add_med=True,
             save=True
             )

# get the most often used words for both positive and negative titles
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist

stop_words = stopwords.words('english')

#add some corpus specific stopwords for the game that will appear all the time
stop_words += ['watch','dogs','game','legion','like','bad','get','else','wd','play','playing']

#tokenizer= TweetTokenizer()
tokenizer = RegexpTokenizer(r'\w+')

def text_process(text):
    tokens = []
    
    for t in text:
        toks = tokenizer.tokenize(t)
        toks = [s.lower() for s in toks if s.lower() not in stop_words]
        
        tokens.extend(toks)
        
    return tokens

pos_tokens = text_process(vader_df['title'][vader_df['compound']>0])
neg_tokens = text_process(vader_df['title'][vader_df['compound']<0])

# get 10 most common words for both positive and negative titles
pos_freq = FreqDist(pos_tokens).most_common(15)
neg_freq = FreqDist(neg_tokens).most_common(15)

# save the positive and negative common words to a file
# just because why not - they can be kept in a 2 column file no problem
pos_words = [w for w,c in pos_freq]
neg_words = [w for w,c in neg_freq]

word_df = pd.DataFrame({'Positive':pos_words,'Negative':neg_words})

word_df.to_csv('pos_neg_words_title.txt',index=False,sep=',')
print('Top positive and negative title words saved')
    












