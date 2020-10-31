# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:36:26 2020

@author: Andrew

Script to take data from a subreddit and save it as a file

Steps for this:
    
    1. Take top fields from desired subreddit
    2. Edit to be able to choose top/new/hot or a topic search on the subreddit
    3. Save the file to a .csv labeled with what we are saving
        Format should be <subreddit>-<sorting method>-<date updated>.csv
    4. .csv should overwrite duplicate topics (same title and author) with new information
    5. .csv should append new data and not completely overwrite the file
"""

