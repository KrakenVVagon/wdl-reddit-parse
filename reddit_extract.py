# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:36:26 2020

@author: Andrew

Script to take data from a subreddit and save it as a file

Steps for this:
    
    1. Take top fields from desired subreddit (done, easy)
    2. Edit to be able to choose top/new/hot or a topic search on the subreddit (done)
    3. Save the file to a .csv labeled with what we are saving (done)
        Format should be <subreddit>-<sorting method>.csv
    4. .csv should overwrite duplicate topics (same title and author) with new information (done)
    5. .csv should append new data and not completely overwrite the file (done)
    
Personal information (psu, key, my login info etc should be stored in a file and read into the script to be safe.)
"""

# import modules
import praw
import pandas as pd

class subreddit:
    
    def __init__(self,subreddit,reddit=None):
        self.reddit = reddit
        
        # if no reddit object is provided just make the reddit out of my metadata
        if self.reddit is None:
            metadata = pd.read_csv('reddit_extra_metadata.txt',delimiter=',',header=None)
            
            self.psu = metadata[1][metadata[0]=='psu'].values[0]
            self.key = metadata[1][metadata[0]=='key'].values[0]
            self.agent= metadata[1][metadata[0]=='agent'].values[0]
            self.user = metadata[1][metadata[0]=='user'].values[0]
            self.password= metadata[1][metadata[0]=='pass'].values[0]
            
            self.reddit = praw.Reddit(client_id=self.psu,
                                      client_secret=self.key,
                                      user_agent=self.agent,
                                      username=self.user,
                                      password=self.password
                                      )
            
        # assign the actual subreddit from praw's api
        self.subreddit = self.reddit.subreddit(subreddit)
        
    # define functions that are needed to parse and save the data
    
    # function to get the posts from the subreddit
    # acceptable values for how are hot, new, search, top
    # default is hot, 1000 limit (highest for reddit API)
    def get_posts(self,how='hot',limit=1000,search=None):
        if how == 'new':
            return self.subreddit.new(limit=limit)
        elif how == 'hot':
            return self.subreddit.hot(limit=limit)
        elif how == 'search':
            if search is None:
                print('No search keywords given')
            return self.subreddit.search(search,limit=limit)
        elif how == 'top':
            return self.subreddit.top(limit=limit)
    
    # function to save the posts and their data in a .csv
    def save_posts(self,how='hot',limit=1000,search=None):
        posts = subreddit.get_posts(self,how=how,limit=limit,search=search)
        path_to_file = '{}-{}.csv'.format(self.subreddit,how)
        
        topics_dict = {'title':[],
                       'score':[],
                       'id':[],
                       'url':[],
                       'commCount':[],
                       'created':[],
                       'author':[],
                       'body':[],
                       'type':[],
                      }
        
        for p in posts:
            topics_dict['title'].append(p.title)
            topics_dict['score'].append(p.score)
            topics_dict['id'].append(p.id)
            topics_dict['url'].append(p.url)
            topics_dict['commCount'].append(p.num_comments)
            topics_dict['created'].append(p.created)
            topics_dict['author'].append(p.author)
            topics_dict['body'].append(p.selftext)
            if p.is_self:
                topics_dict['type'].append('Text')
            else:
                topics_dict['type'].append('Other')
                
        new_data = pd.DataFrame(topics_dict)
        # fix the date column
        import datetime as dt
        
        def get_date(created):
            return dt.datetime.fromtimestamp(created)
    
        _timestamp = new_data['created'].apply(get_date)
        new_data = new_data.assign(timestamp=_timestamp)
        
        # try to read the existing file
        # if no file exists then create the file
        try:
            old_data = pd.read_csv(path_to_file)
        except:
            print('No old file found - creating file {}'.format(path_to_file))
            old_data = None
            print('File saved')
        
        # if there is an existing file update instead of replacing the data
        # take only the unique submission ids to avoid doubling any replies
        if old_data is not None:
            combined = pd.concat([new_data,old_data]).drop_duplicates(subset=['id'])
            combined.to_csv(path_to_file,index=False,mode='w')
            print('File updated')
        else:
            new_data.to_csv(path_to_file,index=False,mode='w')

# going to add another class here where we can take a look at the comments of a reddit post