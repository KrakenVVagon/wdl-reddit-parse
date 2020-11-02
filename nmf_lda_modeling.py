# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:35:04 2020

@author: Andrew

This script have a class that is a corpus of words, and also stop_words for that corpus
It will perform both NMF and LDA topic modeling on the corpus of words and then return the results
"""

import numpy as np
import pandas as pd

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

# class that is the corpus of words
# corpus should be a pandas dataframe of text entries
# stopwords should be a list of stop words. if no list is provided the default english stopwords will be applied
class corpus:
    
    def __init__(self,corpus,stopwords=stopwords.words('english')):
        self.corpus=corpus
        self.stopwords=stopwords
    
    def nmf_topics(self,n,display=False):
        tfidf_vec = TfidfVectorizer(max_df=.95,min_df=1,stop_words=self.stopwords)
        tfidf = tfidf_vec.fit_transform(self.corpus)
        
        # used in the displaying of topics
        tfidf_name = tfidf_vec.get_feature_names()
        
        # number of components is given in NMF topic selection - variable by the user
        nmf = NMF(n_components=n,random_state=13,l1_ratio=.5)
        
        n_fit = nmf.fit(tfidf)
        nmf_output = n_fit.transform(tfidf)
        
        # if displaying the feature names then return the model itself and the feature names
        if display:
            return (n_fit,tfidf_name)
        # otherwise return the NMF output
        else:
            return nmf_output
    
    def lda_topics(self,n,display=False):
        tf_vec = CountVectorizer(max_df=.95,min_df=1,stop_words=self.stopwords)
        tf = tf_vec.fit_transform(self.corpus)
        
        tf_name = tf_vec.get_feature_names()
        
        lda = LatentDirichletAllocation(n_components=n,random_state=13,learning_method='batch')
        l_fit = lda.fit(tf)
        
        lda_output = l_fit.transform(tf)
        
        if display:
            return (l_fit,tf_name)
        else:
            return lda_output
        
    # function to print the n_words top words from the model
    # currently only accepts NMF and LDA as string inputs
    def display_topics(self,model,n,n_words):
        
        if model.upper() == 'NMF':
            fit_model, feature_names = corpus.nmf_topics(self,n,display=True)
        elif model.upper() == 'LDA':
            fit_model, feature_names = corpus.lda_topics(self,n,display=True)
            
        for topic_idx, topic in enumerate(fit_model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_words - 1:-1]]))
    
    def sentiment_analysis(self):
        sia = SIA()
        results = []
        
        for t in self.corpus:
            score = sia.polarity_scores(t)
            results.append(score)
            
        return results
        