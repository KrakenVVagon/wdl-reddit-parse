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
        We also want to compare the score of the reddit posts to the compound score
    3. Topic modeling of larger text posts
        This will be done like the TFIDF work I did for other NLP projects
    4. Sentiment analysis and topic modeling for comments on posts.
        For longer comments perhaps TFIDF is going to be better than VADER for finding sentiment
"""
from reddit_extract import subreddit
import pandas as pd
import os
import time
import numpy as np

wdsr = subreddit('watch_dogs')

# function that checks when the file was last updated
# if it was updated more than 3 days ago run the save_posts process.
# should return the dataframe that is the reddit file
def get_data(subreddit,how='hot'):
    path_to_file = '{}-{}.csv'.format(subreddit.subreddit,how)
    
    # try and check when the file was last modified
    # if the path doesnt exist then make the file
    global updated
    try:
        mod_time = os.path.getmtime(path_to_file)
        cur_time = time.time()
        
        global updated
        
        if (cur_time-mod_time) > 43200:
            updated = True
            print('File out of date - updating')
            subreddit.save_posts()
        else:
            updated = False
        
        df = pd.read_csv(path_to_file)
        
    except:
        updated = True
        subreddit.save_posts()
        df = pd.read_csv(path_to_file)
    
    print('Data loaded')
    return df

def scatter_plot(x,y,title=None,xlabel=None,ylabel=None,size=50,
                 save=False,add_avg=False,add_med=False,ylog=False,xlog=False
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
        
    if ylog:
        plt.yscale('log')
    
    if xlog:
        plt.xscale('log')
    
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
        plt.clf()
    else:
        plt.show()

# calling this function makes the current matplotlib plot pretty.
def pretty_plot(xlabel,ylabel,title,save=False):
    import matplotlib.pyplot as plt

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
        
    if save:
        if title is None:
            title='figure01'
        else:
            title=title.replace(' ','_')
            
        plt.savefig('{}.png'.format(title))
        print('Saved plot to {}.png'.format(title))
        plt.clf()
    

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
    results.append(score)

vader_df = pd.DataFrame(results)
df['title_vader_score'] = vader_df['compound']

yplot = vader_df['compound'][vader_df['compound'] != 0]
xplot = range(len(yplot))

# create scatter plot of vader scores
scatter_plot(x=xplot,
             y=yplot,
             xlabel='Post Number',
             ylabel='Compound VADER Score',
             title='VADER Scores for Watch_Dogs Submission Titles',
             add_avg=True,
             add_med=True,
             save=True
             )

# create scatter plot of vader scores compared to the reddit scores
# which type of posts typically get more attention?
# want the posts that have a score higher than 1 - these get a single downvote and never get seen
xplot = df['title_vader_score'][ (df.score > 1) & (df.title_vader_score != 0) ]
yplot = df['score'][ (df.score > 1) & (df.title_vader_score != 0) ] # reddit score

scatter_plot(x=xplot,
             y=np.log10(yplot),
             xlabel='VADER Score',
             ylabel='Log10 Reddit Score',
             title='VADER Score vs Reddit Score for Watch_Dogs Submission Titles',
             save=True
             )

# that scatter plot did very little - want to box plot the distribution of reddit scores for positive and negative posts

# function to assign positive or negative based on compound scores
def sentiment_result(df,column=''):
    if df[column] > 0:
        return 'Positive'
    elif df[column] < 0:
        return 'Negative'
    else:
        return 'Neutral'
    
df['title_type'] = df.apply(sentiment_result,axis=1,column='title_vader_score')

# density distributions
import seaborn as sns
import matplotlib.pyplot as plt

df_sns = df[['title_vader_score','score','title_type']].dropna()
df_sns = df_sns[df_sns.score > 1]

df_sns.score = np.log10(df_sns.score)

pos_df = df_sns.loc[df_sns.title_type=='Positive']
neg_df = df_sns.loc[df_sns.title_type=='Negative']

plt.figure(figsize=(12,9))
ax1 = sns.kdeplot(pos_df['title_vader_score'],pos_df['score'],
            cmap='Blues',shade=True,shade_lowest=False,n_levels=5
            )

ax2 = sns.kdeplot(neg_df['title_vader_score'],neg_df['score'],
            cmap='Reds',shade=True,shade_lowest=False,n_levels=5
            )

pretty_plot('Title VADER Score','Log10 Reddit Score','VADER Title Score vs Reddit Score Density Plot',save=True)

# get the most often used words for both positive and negative titles
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist

stop_words = stopwords.words('english')

#add some corpus specific stopwords for the game that will appear all the time
stop_words += ['watch','dogs','game','legion','like','bad','get','else','wd','play','playing','watchdogs']

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
    
#### STEP 2 ####
    
################

# take only the posts that are text-based. these should all have content inside their body
text_posts = df[df.type=='Text'].dropna()

# find the average length in characters of one of these text posts
lens = []
for b in text_posts['body']:
    try:
        lens.append(len(b))
    except:
        lens.append(0)
        
lens = np.array(lens)
nonzero_lens = lens[np.nonzero(lens)]
        
print('Average legnth of text posts is: {}'.format(np.mean(lens)))
print('Average length of text posts without 0 length: {}'.format(np.mean(nonzero_lens)))

# good length - on average just under 80 words probably.
# from here we simply do topic analysis using LDA and NMF

from nmf_lda_modeling import corpus

# define the corpus of words - this is the body list from the text_posts
# first is to do sentiment analysis on the corpus and see how often the title and content match
# should user VADER again to be able to match them
texts = corpus(text_posts.body,stopwords=stop_words)

vader_scores = pd.DataFrame(texts.sentiment_analysis())

text_posts['body_vader_score'] = vader_scores['compound']
text_posts['body_type'] = text_posts.apply(sentiment_result,axis=1,column='body_vader_score')

## cross tab for the confusion matrix to see where these agree and where they dont
pred = text_posts['title_type']
expt = text_posts['body_type']

conf_mat = pd.crosstab(expt,pred,rownames=['Body Type'],colnames=['Title Type'],margins=False)
norm_conf= conf_mat / conf_mat.sum(axis=1)

def plot_confusion_matrix(df_confusion, title='VADER Score Confusion Matrix', cmap=plt.cm.gray_r,save=False):
    plt.figure(figsize=(12,9))
    plt.matshow(df_confusion, cmap=cmap,fignum=1) # imshow
    #plt.title(title,fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45,fontsize=14)
    plt.yticks(tick_marks, df_confusion.index,fontsize=14)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name,fontsize=16)
    plt.xlabel(df_confusion.columns.name,fontsize=16)
    
    if save:
        if title is None:
            title='figure01'
        else:
            title=title.replace(' ','_')
            
        plt.savefig('{}.png'.format(title))
        print('Saved plot to {}.png'.format(title))
        plt.clf()
    else:
        plt.show()

plot_confusion_matrix(norm_conf,save=True)

# dont have good agreement betwene the sentiment analysis for the titles vs. the bodies
# we basically assume that the bodies are going to be better since they have more text
# there are way too many neutral titles - very few neutral bodies in comparison

# make the same plots for VADER scores for body text as for submission titles
yplot = text_posts['body_vader_score'][text_posts['body_vader_score'] != 0]
yplot = yplot.dropna()
xplot = range(len(yplot))

# create scatter plot of vader scores
scatter_plot(x=xplot,
             y=yplot,
             xlabel='Post Number',
             ylabel='Compound VADER Score',
             title='VADER Scores for Watch_Dogs Text Posts',
             add_avg=True,
             add_med=True,
             save=True
             )

# density plot of vader scores and reddit scores for text posts
df_sns = text_posts[['body_vader_score','score','body_type']].dropna()
df_sns = df_sns[df_sns.score > 1]

df_sns.score = np.log10(df_sns.score)

pos_df = df_sns.loc[df_sns.body_type=='Positive']
neg_df = df_sns.loc[df_sns.body_type=='Negative']

plt.figure(figsize=(12,9))
ax1 = sns.kdeplot(pos_df['body_vader_score'],pos_df['score'],
            cmap='Blues',shade=True,shade_lowest=False,n_levels=5
            )

ax2 = sns.kdeplot(neg_df['body_vader_score'],neg_df['score'],
            cmap='Reds',shade=True,shade_lowest=False,n_levels=5
            )

pretty_plot('Body VADER Score','Log10 Reddit Score','VADER Body Score vs Reddit Score Density Plot',save=True)

# take the positive bodies and look and their topics
# need to do some pre-processing on this data
# most notably removing the URL text
import re
text_posts['cleanBody'] = text_posts['body'].apply(lambda x: re.sub('http\S+|www.\S+',' ',str(x)))

# also apply the redditcleaner module
import redditcleaner
text_posts['cleanBody'] = text_posts['cleanBody'].apply(lambda x: redditcleaner.clean(x))

text_posts['cleanBody'] = text_posts['cleanBody'].apply(lambda x: x.encode('ascii','ignore').decode('utf-8'))
text_posts['cleanBody'] = text_posts['cleanBody'].apply(lambda x: x.replace('x200b',' '))

pos_df = text_posts.loc[text_posts.body_type=='Positive']

# add other stop words that we definitely need
stop_words += ['anyone','one','know','got','even','go',
               'really','would','could','im','dont',
               'title','says','question','x200b',
               'cant','ive','find','looking','way','want','still',
               'wd2','first','much','spoilers','poll'
               ]
pos_corpus = corpus(pos_df['cleanBody'],stopwords=stop_words)

# NMF machine learning proves stronger than LDA again

# topic0 talks about characters and recruiting - spies, hitmen etc
# topic1 talks about villains, 404 missions
# topic2 talks about operatives, permadeath, saves, progress
# topic3 talks about drones, cargo drones, construction workers, spiderbot

if updated:
    print('FILE UPDATED - REVIEW TOPICS')
    pos_corpus.display_topics('NMF',4,10)
else:
    nmf_vals = pos_corpus.nmf_topics(4)
    nmf_df = pd.DataFrame(nmf_vals,columns=['topic0','topic1','topic2','topic3'])
    #nmf_df['post_number']=nmf_df.index
   # nmf_df = nmf_df.replace(0,np.nan,inplace=True)
    
    #melt_df = nmf_df.melt(var_name='groups',value_name='vals')
    
    #sns.violinplot(data=nmf_df,scale='count',inner=None)
    fig, axes = plt.subplots(2,2,figsize=(12,9),sharey=True)
    axes[0,0].set_title('Topic0 NMF Scores',fontsize=16)
    axes[0,1].set_title('Topic1 NMF Scores',fontsize=16)
    axes[1,0].set_title('Topic2 NMF Scores',fontsize=16)
    axes[1,1].set_title('Topic3 NMF Scores',fontsize=16)
    
    sns.violinplot(y=nmf_df['topic0'][nmf_df.topic0 > 0],ax=axes[0,0],inner=None,color='blue')
    sns.violinplot(y=nmf_df['topic1'][nmf_df.topic1 > 0],ax=axes[0,1],inner=None,color='red')
    sns.violinplot(y=nmf_df['topic2'][nmf_df.topic2 > 0],ax=axes[1,0],inner=None,color='green')
    sns.violinplot(y=nmf_df['topic3'][nmf_df.topic3 > 0],ax=axes[1,1],inner=None,color='orange')
    
    for ax in axes.flat:
        ax.set(ylabel='Score')
    plt.savefig('positive_body_NMF_scores.png')
    plt.clf()

## repeat this process for negative comments

neg_df = text_posts.loc[text_posts.body_type=='Negative']

# update stop words with new findings
stop_words += ['thing','something','feel','think','edit','say',
               'ubisoft','games','see','far'
               ]

neg_corpus = corpus(neg_df['cleanBody'],stopwords=stop_words)

# topic0 talks about gameplay - characters, driving and repetition and permadeath
# topic1 talks about maps and UI
# topic2 talks about performance issues (PC) and crashes/progress
# topic3 talks about recruitment and operatives, deep profiler, repetition

if updated:
    print('FILE UPDATED - REVIEW TOPICS')
    neg_corpus.display_topics('NMF',4,10)
else:
    nmf_vals = neg_corpus.nmf_topics(4)
    nmf_df = pd.DataFrame(nmf_vals,columns=['topic0','topic1','topic2','topic3'])
    #nmf_df['post_number']=nmf_df.index
   # nmf_df = nmf_df.replace(0,np.nan,inplace=True)
    
    #melt_df = nmf_df.melt(var_name='groups',value_name='vals')
    
    #sns.violinplot(data=nmf_df,scale='count',inner=None)
    fig, axes = plt.subplots(2,2,figsize=(12,9),sharey=True)
    axes[0,0].set_title('Topic0 NMF Scores',fontsize=16)
    axes[0,1].set_title('Topic1 NMF Scores',fontsize=16)
    axes[1,0].set_title('Topic2 NMF Scores',fontsize=16)
    axes[1,1].set_title('Topic3 NMF Scores',fontsize=16)
    
    sns.violinplot(y=nmf_df['topic0'][nmf_df.topic0 > 0],ax=axes[0,0],inner=None,color='blue')
    sns.violinplot(y=nmf_df['topic1'][nmf_df.topic1 > 0],ax=axes[0,1],inner=None,color='red')
    sns.violinplot(y=nmf_df['topic2'][nmf_df.topic2 > 0],ax=axes[1,0],inner=None,color='green')
    sns.violinplot(y=nmf_df['topic3'][nmf_df.topic3 > 0],ax=axes[1,1],inner=None,color='orange')
    
    for ax in axes.flat:
        ax.set(ylabel='Score')
    plt.savefig('negative_body_NMF_scores.png')
    plt.clf()











