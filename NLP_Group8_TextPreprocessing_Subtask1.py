#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import nltk
import string

def textpreprocess():
    #Open Training Subtask 1 CSV
    #RETURN Array of entries 
    #Per Entry: user_ID,text_ID, timestamp, collection_phase,is_words(TRUE/FALSE), Valence(-2.0 to 2.0), Arousal(-1.0 to 1.0), Token List
    
    #Load CSV file
    with open('train_subtask1.csv','r',newline='') as subtask1csv:
        reader = csv.reader(subtask1csv)
        #Remove CSV header
        next(reader)
        subtask1array = list(reader)
    
    tokenized_task1 = []

    for row in subtask1array:

        #Preprocess Text
        text = row[2]
        #Lowercase
        text = text.lower()
        #Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        #Append data for entry, tokenized text at the end
        entry = [(row[0]),(row[1]),(row[3]),(row[4]),(row[5]),(row[6]),(row[7]),nltk.tokenize.word_tokenize(text)]
        tokenized_task1.append(entry)
    
    return tokenized_task1

