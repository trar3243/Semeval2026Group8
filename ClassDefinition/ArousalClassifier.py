import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta

class AffectClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension_size = Roberta.output_dimension_size + 1 # 1 is for is_words  
        self.hidden_dim = self.input_dimension_size//2
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 10% of the hidden layer nodes are dropped 
        )
        self.arousal_layer = torch.nn.Linear(self.hidden_dim, 3) 
        self.valence_layer = torch.nn.Linear(self.hidden_dim, 5) 

        self.__initialize_weights__()
    def __initialize_weights__(self):
        for module in [self.shared, self.arousal_layer, self.valence_layer]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    def forward(self, features: torch.Tensor):
        shared = self.shared(features)
        arousal_classification = self.arousal_layer(shared)
        valence_classification = self.valence_layer(shared)
        return {
            "arousal_logits": arousal_classification, # these are logits 
            "valence_logits": valence_classification
        }
