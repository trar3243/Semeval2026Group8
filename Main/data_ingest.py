#!../bin/python3
import pandas
import csv
import os, sys 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Entry import Entry

def ingest(dataPath):
    
    #Objects for each CSV row, append to entries
    entries = []
    with open(dataPath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                #New object per row
                e = Entry()

                # Assign everything as-is
                e.user_id = row.get('user_id')
                e.text_id = row.get('text_id')
                e.text = row.get('text')
                e.timestamp = row.get('timestamp')
                e.collection_phase = row.get('collection_phase')
                e.is_words = row.get('is_words')
                e.valence = float(row.get('valence'))
                e.arousal = float(row.get('arousal'))
                #Append this csv row's data to entries
                entries.append(e)
    #Return all
    return entries
