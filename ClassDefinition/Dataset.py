import sys, os
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
        self.arousalLabelList = torch.FloatTensor(self.arousalLabelList).to(torch.long)
    def __init_valence_label_list__(self):
        self.valenceLabelList = []
        for entry in self.entryList:
            self.valenceLabelList.append(entry.valence_class)
        self.valenceLabelList = torch.FloatTensor(self.valenceLabelList).to(torch.long)
    def getFeatures(self): # where a lot of time will be taken 
        # creates its own Roberta() and computes embeddings every time a batch is fetched meaning  inside the training loop
        # -> very slow, no caching of embeddings between epochs
        # TODO might want to update later on if needed
        featureList=[]
        for entry in self.entryList:
            self.roberta.setText(entry.text)
            singleTextEmbedding = self.roberta.getClsEmbedding()[0]
            featureList.append(torch.FloatTensor(singleTextEmbedding))
        return torch.stack(featureList)# the feature list is not being stored currently because could take up RAM. Can choose to change this later on.
    def getLabels(self):
        return self.arousalLabelList 


class Dataset:
    def __init__(self, entryList, roberta):
        self.roberta = roberta
        self.trainSet, self.devSet = train_test_split(entryList, test_size=0.2, random_state=42)
        self.trainBatchList = None
    def shuffle(self):
        random.shuffle(self.trainSet)
        random.shuffle(self.devSet) # shouldnt matter
    def setTrainBatchList(self,batchSize):
        self.trainBatchList = []
        for i in range(0, len(self.trainSet), batchSize):
            self.trainBatchList.append(Batch(self.trainSet[i:i+batchSize], self.roberta))
    def getTrainBatchList(self):
        return self.trainBatchList
