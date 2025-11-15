import sys, os
import torch 
import random 
from sklearn.model_selection import train_test_split
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta

class Batch:
    def __init__(self,entryList):
        self.entryList=entryList
        self.__init_arousal_label_list__()
        self.__init_valence_label_list__()
    def __init_arousal_label_list__(self):
        self.arousalLabelList = []
        for entry in self.entryList:
            self.arousalLabelList.append(entry.arousal_class)
        self.arousalLabelList = torch.FloatTensor(self.arousalLabelList).to(torch.long)
    def __init_valence_label_list__(self):
        self.valenceLabelList = []
        for entry in self.entryList:
            self.valenceLabelList.append(entry.valence_class)
        self.valenceLabelList = torch.FloatTensor(self.valenceLabelList).to(torch.long)
    def getFeatures(self): # where a lot of time will be taken 
        featureList=[]
        roberta=Roberta()
        for entry in self.entryList:
            roberta.setText(entry.text)
            singleTextEmbedding = roberta.getClsEmbedding()[0]
            featureList.append(torch.FloatTensor(singleTextEmbedding))
        return torch.stack(featureList)# the feature list is not being stored currently because could take up RAM. Can choose to change this later on.
    def getLabels(self):
        return self.arousalLabelList 


class Dataset:
    def __init__(self,entryList):
        self.trainSet, self.devSet = train_test_split(entryList, test_size=0.2, random_state=42)
        self.trainBatchList=None
    def shuffle(self):
        random.shuffle(self.trainSet)
        random.shuffle(self.devSet) # shouldnt matter
    def setTrainBatchList(self,batchSize):
        self.trainBatchList = []
        for i in range(0, len(self.trainSet), batchSize):
            self.trainBatchList.append(Batch(self.trainSet[i:i+batchSize]))
    def getTrainBatchList(self):
        return self.trainBatchList
