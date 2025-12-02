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
        self.valence = float(row.get('valence')) # -2,...,2
        self.arousal = float(row.get('arousal')) # 0,...,2
        self.user_id_index = None # gets assigned in the Dataset class 
        
        self.mean_lexical_valence = None
        self.mean_lexical_arousal = None
        self.count_lexical_high_valence = None
        self.count_lexical_low_valence = None
        self.count_lexical_high_arousal = None
        self.count_lexical_low_arousal = None

        self.__set_valence_class__()
        self.__set_arousal_class__()
    def __set_valence_class__(self):
        valence_class = self.valence # -2,...,2
        valence_class = max(-2.0, min(2.0, valence_class))
        self.valence_class = valence_class# -2,...,2
    def __set_arousal_class__(self):
        arousal_class = self.arousal - 1.0 # -1,...,1
        arousal_class = max(-1.0, min(1.0, arousal_class))
        self.arousal_class = arousal_class# 0,...,2
    def __repr__(self):
        return (
            f"Entry(user_id={self.user_id}, text_id={self.text_id}, text={self.text}, timestamp={self.timestamp}, collection_phase={self.collection_phase}, is_words={self.is_words}, valence={self.valence}, arousal={self.arousal}, valence_class={self.valence_class}, arousal_class={self.arousal_class}, mean_lexical_valence={self.mean_lexical_valence},count_lexical_high_valence={self.count_lexical_high_valence},count_lexical_low_valence={self.count_lexical_low_valence},mean_lexical_arousal={self.mean_lexical_arousal},count_lexical_high_arousal={self.count_lexical_high_arousal},count_lexical_low_arousal={self.count_lexical_low_arousal},)")
