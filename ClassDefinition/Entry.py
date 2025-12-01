import csv
import os, sys 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Utils import convertStringToFloat 

class Entry:
    def __init__(self, row):
        self.user_id = int(row.get('user_id'))
        self.text_id = int(row.get('text_id'))
        self.text = row.get('text')
        self.timestamp = row.get('timestamp')
        self.collection_phase = row.get('collection_phase')
        self.is_words = convertStringToFloat(row.get('is_words'))
        self.valence = int(float(row.get('valence'))) # -2,...,2
        self.arousal = int(float(row.get('arousal'))) # 0,...,2
        self.user_id_index = None # gets assigned in the Dataset class 

        self.__set_valence_class__()
        self.__set_arousal_class__()
    def __set_valence_class__(self):
        valence_class = self.valence + 2 # 0,...,4
        self.valence_class = max(0, min(4, valence_class))
    def __set_arousal_class__(self):
        arousal_class = self.arousal # 0,...,2
        self.arousal_class = max(0, min(2, arousal_class))
    def __repr__(self):
        return (
            f"Entry(user_id={self.user_id}, text_id={self.text_id}, text={self.text}, timestamp={self.timestamp}, collection_phase={self.collection_phase}, is_words={self.is_words}, valence={self.valence}, arousal={self.arousal}, valence_class={self.valence_class}, arousal_class={self.arousal_class})"
        )
