import sys, os
from collections import Counter
import numpy as np
import re 
import csv 
import torch 
import random 
from sklearn.model_selection import train_test_split
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta
from ClassDefinition.Utils import Logger, ArgumentParser
g_Logger = Logger(__name__)
print = g_Logger.print

class Batch:
    def __init__(self, entryList, roberta):
        self.entryList = entryList
        self.roberta = roberta
        self.__init_arousal_label_list__()
        self.__init_valence_label_list__()
    
    def __init_arousal_label_list__(self):
        self.arousalLabelList = []
        for entry in self.entryList:
            self.arousalLabelList.append(entry.arousal_class)
        self.arousalLabelList = torch.tensor(self.arousalLabelList, dtype=torch.long)
    
    def __init_valence_label_list__(self):
        self.valenceLabelList = []
        for entry in self.entryList:
            self.valenceLabelList.append(entry.valence_class)
        self.valenceLabelList = torch.tensor(self.valenceLabelList, dtype=torch.long)
    
    def getClsEmbeddings(self):  
        self.roberta.setTextList([e.text for e in self.entryList])
        return self.roberta.getClsEmbedding() # b, 768 
    def getUserIndices(self):
        return torch.tensor([e.user_id_index for e in self.entryList], dtype=torch.long)
    def getIsWords(self):
        return torch.tensor([e.is_words for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    
    def getMeanLexicalValence(self):
        return torch.tensor([e.mean_lexical_valence for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    def getMeanLexicalArousal(self):
        return torch.tensor([e.mean_lexical_arousal for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    def getCountLexicalHighValence(self):
        return torch.tensor([e.count_lexical_high_valence for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    def getCountLexicalLowValence(self):
        return torch.tensor([e.count_lexical_low_valence for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    def getCountLexicalHighArousal(self):
        return torch.tensor([e.count_lexical_high_arousal for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    def getCountLexicalLowArousal(self):
        return torch.tensor([e.count_lexical_low_arousal for e in self.entryList], dtype = torch.float32).unsqueeze(1)
        
        

class LexicalValenceArousal:
    def __init__(self, row, isEmpty = False):
        if(isEmpty):
            self.word = None
            self.mean_valence = 0
            self.mean_arousal = 0
            self.is_high_valence = False
            self.is_low_valence = False
            self.is_high_arousal = False
            self.is_low_arousal = False

        else:
            self.word = row.get('Word')
            self.mean_valence = float(row.get('V.Mean.Sum'))
            self.mean_arousal = float(row.get('A.Mean.Sum'))
            self.is_high_valence = self.mean_valence > 7
            self.is_low_valence = self.mean_valence < 3
            self.is_high_arousal = self.mean_arousal > 7
            self.is_low_arousal = self.mean_arousal < 3
        

class Dataset:
    def __init__(self, dataPath, lexiconLookupPath, roberta:Roberta):
        self.__set_entry_list__(dataPath)
        self.__set_user_indices__()
        self.__set_valence_arousal_lexicon_features__(lexiconLookupPath)
        self.roberta = roberta
        self.trainSet, self.devSet = train_test_split(self.entryList, test_size=0.2, random_state=42)
        self.trainBatchList = None
        self.devBatchList = None

    def __set_entry_list__(self, dataPath):
        self.entryList = []
        with open(dataPath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.entryList.append(Entry(row))

    def __set_valence_arousal_lexicon_features__(self, lookupPath):
        word_dict = {}
        with open(lookupPath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lexicalValenceArousal = LexicalValenceArousal(row)
                word_dict[lexicalValenceArousal.word] = lexicalValenceArousal
        for entry in self.entryList:
            sum_positive_valence = 0 
            sum_negative_valence = 0 
            mean_valence = 0
            sum_positive_arousal = 0 
            sum_negative_arousal = 0 
            mean_arousal = 0
            counter = 0 
            words = re.findall(r"\b\w+\b", entry.text.lower())
            for word in words:
                lexicalValenceArousal = word_dict.get(word, LexicalValenceArousal(None, isEmpty=True))
                sum_positive_valence = sum_positive_valence + int(lexicalValenceArousal.is_high_valence)
                sum_negative_valence = sum_negative_valence + int(lexicalValenceArousal.is_low_valence)
                mean_valence = mean_valence + lexicalValenceArousal.mean_valence
                sum_positive_arousal = sum_positive_arousal + int(lexicalValenceArousal.is_high_arousal)
                sum_negative_arousal = sum_negative_arousal + int(lexicalValenceArousal.is_low_arousal)
                mean_arousal = mean_arousal + lexicalValenceArousal.mean_arousal
                if(lexicalValenceArousal.word is not None):
                    counter = counter + 1
            mean_valence = mean_valence / (counter if counter > 0 else 1) 
            mean_arousal = mean_arousal / (counter if counter > 0 else 1)
            
            entry.mean_lexical_valence = mean_valence
            entry.mean_lexical_arousal = mean_arousal 
            entry.count_lexical_high_valence = sum_positive_valence
            entry.count_lexical_low_valence = sum_negative_valence
            entry.count_lexical_high_arousal = sum_positive_arousal
            entry.count_lexical_low_arousal = sum_negative_arousal
            
        total_mean_lexical_valence = np.mean([e.mean_lexical_valence for e in self.entryList])
        total_mean_lexical_arousal = np.mean([e.mean_lexical_arousal for e in self.entryList])
        total_mean_count_lexical_high_valence = np.mean([e.count_lexical_high_valence for e in self.entryList])
        total_mean_count_lexical_low_valence = np.mean([e.count_lexical_low_valence for e in self.entryList])
        total_mean_count_lexical_high_arousal = np.mean([e.count_lexical_high_arousal for e in self.entryList])
        total_mean_count_lexical_low_arousal = np.mean([e.count_lexical_low_arousal for e in self.entryList])
        
        total_std_lexical_valence = np.std([e.mean_lexical_valence for e in self.entryList])
        total_std_lexical_arousal = np.std([e.mean_lexical_arousal for e in self.entryList])
        total_std_count_lexical_high_valence = np.std([e.count_lexical_high_valence for e in self.entryList])
        total_std_count_lexical_low_valence = np.std([e.count_lexical_low_valence for e in self.entryList])
        total_std_count_lexical_high_arousal = np.std([e.count_lexical_high_arousal for e in self.entryList])
        total_std_count_lexical_low_arousal = np.std([e.count_lexical_low_arousal for e in self.entryList])

        for entry in self.entryList: 
            entry.mean_lexical_valence = (entry.mean_lexical_valence - total_mean_lexical_valence)/ total_std_lexical_valence
            entry.mean_lexical_arousal = (entry.mean_lexical_arousal -  total_mean_lexical_arousal)/ total_std_lexical_arousal 
            entry.count_lexical_high_valence = (entry.count_lexical_high_valence - total_mean_count_lexical_high_valence)/ total_std_count_lexical_high_valence
            entry.count_lexical_low_valence = (entry.count_lexical_low_valence - total_mean_count_lexical_low_valence)/ total_std_count_lexical_low_valence 
            entry.count_lexical_high_arousal = (entry.count_lexical_high_arousal - total_mean_count_lexical_high_arousal)/ total_std_count_lexical_high_arousal 
            entry.count_lexical_low_arousal = (entry.count_lexical_low_arousal -  total_mean_count_lexical_low_arousal)/ total_std_count_lexical_low_arousal 
            


                

    def __set_user_indices__(self):
        user_id_list = [e.user_id for e in self.entryList]
        user_id_distinct_ordered_list = sorted(list(set(user_id_list)))
        self.number_of_users = len(user_id_distinct_ordered_list)
        id_to_index = {uid:i for i,uid in enumerate(user_id_distinct_ordered_list)}
        for entry in self.entryList:
            entry.user_id_index = id_to_index[entry.user_id]
            
    def shuffle(self):
        random.shuffle(self.trainSet)
    def setTrainBatchList(self,batchSize):
        self.trainBatchList = []
        for i in range(0, len(self.trainSet), batchSize):
            self.trainBatchList.append(Batch(self.trainSet[i:i+batchSize], self.roberta))
    def setDevBatchList(self,batchSize):
        self.devBatchList = [] 
        for i in range(0, len(self.devSet), batchSize):
            self.devBatchList.append(Batch(self.devSet[i:i+batchSize], self.roberta))
    def getDevBatchList(self):
        return self.devBatchList
    def getTrainBatchList(self):
        return self.trainBatchList
    def printSetDistribution(self):
        valence_class_counts = Counter(item.valence_class for item in self.entryList)
        arousal_class_counts = Counter(item.arousal_class for item in self.entryList)
        print(f"EntryList has {len(self.entryList)} entries, {self.number_of_users} distinct users with following label distribution:\n\tvalence:{valence_class_counts}\n\tarousal:{arousal_class_counts}")
