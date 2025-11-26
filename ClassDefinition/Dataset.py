import sys, os
from collections import Counter
import torch 
import random 
from sklearn.model_selection import train_test_split
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta

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
    
    def getFeatures(self):  
        self.roberta.setTextList([e.text for e in self.entryList])
        return self.roberta.getClsEmbedding()


class Dataset:
    def __init__(self, entryList, roberta:Roberta):
        self.roberta = roberta
        self.trainSet, self.devSet = train_test_split(entryList, test_size=0.2, random_state=42)
        self.trainBatchList = None
        self.devBatchList = None
        self.__set_valence_arousal_means__()
    def __set_valence_arousal_means__(self):
        full_set = self.trainSet + self.devSet 
        self.valence_mean = sum([item.valence_class for item in full_set]) / len(full_set)
        self.arousal_mean = sum([item.arousal_class for item in full_set]) / len(full_set)
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
        full_set = self.trainSet + self.devSet
        valence_class_counts = Counter(item.valence_class for item in full_set)
        arousal_class_counts = Counter(item.arousal_class for item in full_set)
        print(f"EntryList has following distribution:\n\tvalence:{valence_class_counts}\n\tarousal:{arousal_class_counts}") 
        print(f"EntryList has following means:\n\tvalence:{self.valence_mean}\n\tarousal:{self.arousal_mean}") 
