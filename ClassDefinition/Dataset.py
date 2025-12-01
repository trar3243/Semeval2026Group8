import sys, os
from collections import Counter
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
        self.arousalLabelList = torch.tensor(self.arousalLabelList, dtype=torch.float)
    
    def __init_valence_label_list__(self):
        self.valenceLabelList = []
        for entry in self.entryList:
            self.valenceLabelList.append(entry.valence_class)
        self.valenceLabelList = torch.tensor(self.valenceLabelList, dtype=torch.float)
    
    def getClsEmbeddings(self):  
        self.roberta.setTextList([e.text for e in self.entryList])
        return self.roberta.getClsEmbedding() # b, 768 
    def getUserIndices(self):
        return torch.tensor([e.user_id_index for e in self.entryList], dtype=torch.long)
    def getIsWords(self):
        return torch.tensor([e.is_words for e in self.entryList], dtype = torch.float32)


class Dataset:
    def __init__(self, dataPath, roberta:Roberta):
        self.__set_entry_list__(dataPath)
        self.__set_user_indices__()
        self.roberta = roberta
        self.trainSet, self.devSet = train_test_split(self.entryList, test_size=0.2, random_state=42)
        self.trainBatchList = None
        self.devBatchList = None
        self.__set_valence_arousal_means__()
    def __set_valence_arousal_means__(self):
        self.valence_mean = sum([item.valence_class for item in self.entryList]) / len(self.entryList)
        self.arousal_mean = sum([item.arousal_class for item in self.entryList]) / len(self.entryList)

    def __set_entry_list__(self, dataPath):
        self.entryList = []
        with open(dataPath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.entryList.append(Entry(row))

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
        print(f"EntryList has following means:\n\tvalence:{self.valence_mean}\n\tarousal:{self.arousal_mean}") 
