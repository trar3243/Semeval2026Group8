import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta

class ArousalClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension_size = Roberta.output_dimension_size 
        #self.regressor = torch.nn.Linear(768, 1)
        self.hidden_dim = self.input_dimension_size//2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 10% of the hidden layer nodes are dropped 
            torch.nn.Linear(self.hidden_dim, 1) # then go from hidden to single output 
        )
    def __initialize_weights__(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    def forward(self, features: torch.Tensor):
        return self.mlp(features).squeeze(-1)

class ValenceClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension_size = Roberta.output_dimension_size 
        #self.regressor = torch.nn.Linear(768, 1)
        self.hidden_dim = self.input_dimension_size//2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 10% of the hidden layer nodes are dropped 
            torch.nn.Linear(self.hidden_dim, 1) # then go from hidden to single output 
        )
    def __initialize_weights__(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    def forward(self, features: torch.Tensor):
        return self.mlp(features).squeeze(-1)

class AffectClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension_size = Roberta.output_dimension_size 
        self.hidden_dim = self.input_dimension_size//2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 10% of the hidden layer nodes are dropped 
            torch.nn.Linear(self.hidden_dim, 2) # then go from hidden to TWO outputs 
        )
    def __initialize_weights__(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    def forward(self, features: torch.Tensor):
        return torch.sigmoid(self.mlp(features)) # currently using sigmoid 
