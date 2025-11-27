import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta

class LearnableSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        return torch.sigmoid(self.alpha * x + self.beta)

class AffectClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension_size = Roberta.output_dimension_size + 1 # 1 is for is_words  
        self.hidden_dim = self.input_dimension_size//2
        self.sigmoid = LearnableSigmoid()
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
        return self.sigmoid(self.mlp(features)) # currently using sigmoid 
