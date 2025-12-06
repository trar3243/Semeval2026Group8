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
    def __init__(self, entryList, robertaA, robertaB, robertaD, robertaG, robertaH):
        self.entryList = entryList
        self.robertaA = robertaA
        self.robertaB = robertaB
        self.robertaD = robertaD
        self.robertaG = robertaG
        self.robertaH = robertaH
        self.__init_arousal_label_list__()
        self.__init_valence_label_list__()
    
    def __init_arousal_label_list__(self):
        self.arousalLabelList = []
        for entry in self.entryList:
            self.arousalLabelList.append(entry.arousal_class)
        self.arousalLabelList = torch.tensor(self.arousalLabelList, dtype=torch.float32)
    
    def __init_valence_label_list__(self):
        self.valenceLabelList = []
        for entry in self.entryList:
            self.valenceLabelList.append(entry.valence_class)
        self.valenceLabelList = torch.tensor(self.valenceLabelList, dtype=torch.float32)
    
    def getClsEmbeddingsA(self):  
        self.robertaA.setTextList([e.text for e in self.entryList])
        return self.robertaA.getClsEmbedding() # b, 768 
    def getClsEmbeddingsB(self):  
        self.robertaB.setTextList([e.text for e in self.entryList])
        return self.robertaB.getClsEmbedding() # b, 768 
    def getClsEmbeddingsD(self):  
        self.robertaD.setTextList([e.text for e in self.entryList])
        return self.robertaD.getClsEmbedding() # b, 768 
    def getClsEmbeddingsG(self):  
        self.robertaG.setTextList([e.text for e in self.entryList])
        return self.robertaG.getClsEmbedding() # b, 768 
    def getClsEmbeddingsH(self):
        self.robertaH.setTextList([e.text for e in self.entryList])
        return self.robertaH.getClsEmbedding() # b, 768
    def getUserIndices(self):
        return torch.tensor([e.user_id_index for e in self.entryList], dtype=torch.long)
    def getIsWords(self):
        return torch.tensor([e.is_words for e in self.entryList], dtype = torch.float32).unsqueeze(1)
    


class Dataset:
    def __init__(
        self, 
        dataPath, 
        robertaA, robertaB, robertaD, robertaG, robertaH,
        eval_mode, devSetPath, trainSetPath 
    ):
        self.dataPath = dataPath
        self.devSetPath=devSetPath
        self.trainSetPath=trainSetPath
        self.robertaA = robertaA
        self.robertaB = robertaB
        self.robertaD = robertaD
        self.robertaG = robertaG
        self.robertaH = robertaH
        
        if(eval_mode == False):
            self.entryList = self.__get_entry_list__(self.dataPath)
            self.trainSet, self.devSet = train_test_split(self.entryList, test_size=0.1, random_state=42)
        else:
            self.devSet = self.__get_entry_list__(self.devSetPath) 
            self.trainSet = self.__get_entry_list__(self.trainSetPath) 
            self.entryList = self.devSet + self.trainSet
        
        self.__set_user_indices__()

        self.train_user_ids = {e.user_id_index for e in self.trainSet}
        self.dev_user_ids = {e.user_id_index for e in self.devSet}
        
        self.trainBatchList = None
        self.devBatchList = None

    def __get_entry_list__(self, dataPath):
        entryList = []
        with open(dataPath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                entryList.append(Entry(row))
        # entryList = entryList[:20]
        return entryList

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
            self.trainBatchList.append(Batch(self.trainSet[i:i+batchSize], self.robertaA, self.robertaB, self.robertaD, self.robertaG, self.robertaH))
    def setDevBatchList(self,batchSize):
        self.devBatchList = [] 
        for i in range(0, len(self.devSet), batchSize):
            self.devBatchList.append(Batch(self.devSet[i:i+batchSize], self.robertaA, self.robertaB, self.robertaD, self.robertaG, self.robertaH))
    def getDevBatchList(self):
        return self.devBatchList
    def getTrainBatchList(self):
        return self.trainBatchList
    def printSetDistribution(self):
        valence_class_counts = Counter(item.valence_class for item in self.entryList)
        arousal_class_counts = Counter(item.arousal_class for item in self.entryList)
        print(f"EntryList has {len(self.entryList)} entries, {self.number_of_users} distinct users with following label distribution:\n\tvalence:{valence_class_counts}\n\tarousal:{arousal_class_counts}")
    def __convert_float_to_bool__(self, is_words):
        if(is_words == 1.0):
            return "True"
        elif(is_words == 0.0):
            return "False"
        raise Exception(f"{is_words} not in [0.0,1.0]")
    def writeOutSet(self, entrySet, path):
        rows = []
        for entry in entrySet:
            rows.append({
                "user_id":entry.user_id, 
                "text_id":entry.text_id,
                "text":entry.text, 
                "timestamp":entry.timestamp, 
                "collection_phase":entry.collection_phase, 
                "is_words":self.__convert_float_to_bool__(entry.is_words),
                "valence":entry.valence,
                "arousal":entry.arousal
            })
        fieldnames = ["user_id", "text_id", "text", "timestamp", "collection_phase", "is_words", "valence", "arousal"]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)
            f.close()

    def writeOutDevSet(self):
        self.writeOutSet(self.devSet, self.devSetPath)
        print(f"Completed write out of dev set to {self.devSetPath}")
    def writeOutTrainSet(self):
        self.writeOutSet(self.trainSet, self.trainSetPath)
        print(f"Completed write out of train set to {self.trainSetPath}")
        
