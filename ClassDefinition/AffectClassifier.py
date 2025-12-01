import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta

class AffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4 # 1 is for is_words  
        self.norm = torch.nn.LayerNorm(self.input_dimension_size) 
        self.hidden_dim = self.input_dimension_size//2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 10% of the hidden layer nodes are dropped 
            torch.nn.Linear(self.hidden_dim, 2) # then go from hidden to TWO outputs 
        )
        self.__initialize_weights__()
    def __initialize_weights__(self):
        for module in [self.mlp, self.user_embedding]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    def forward(self, cls_embeddings: torch.Tensor, user_indices: torch.Tensor, is_word_indices: torch.Tensor):
        user_matrix = self.user_embedding(user_indices)
        full_feature_matrix = torch.cat([cls_embeddings, user_matrix, is_word_indices.unsqueeze(1)], dim=1)
        full_feature_matrix = self.norm(full_feature_matrix)
        return torch.sigmoid(self.mlp(full_feature_matrix)) # currently using sigmoid 
