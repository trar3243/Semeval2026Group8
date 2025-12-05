import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta


class VersionAAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4 # 1 is for is_words  
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
    def forward(self, 
                cls_embeddings: torch.Tensor, 
                user_indices: torch.Tensor, 
                is_word_indices: torch.Tensor
    ):
        user_matrix = self.user_embedding(user_indices)
        full_feature_matrix = torch.cat([
            cls_embeddings, 
            user_matrix, 
            is_word_indices
        ], dim=1)
        # full_feature_matrix = self.norm(full_feature_matrix)
        logits = self.mlp(full_feature_matrix)
        probs = torch.sigmoid(logits)
        valence_probs = probs[:,0]
        arousal_probs = probs[:,1]
        return {
            "arousal": arousal_probs,
            "valence": valence_probs
        }

class VersionBAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):# , valence_mean, arousal_mean):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4# 1 is for is_words  
        # self.norm = torch.nn.LayerNorm(self.input_dimension_size) 
        self.hidden_dim = self.input_dimension_size//2
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 10% of the hidden layer nodes are dropped 
            torch.nn.Linear(self.hidden_dim, 2) # then go from hidden to TWO outputs 
        )
        self.__initialize_weights__() #valence_mean, arousal_mean)
    def __initialize_weights__(self): #, valence_mean, arousal_mean):
        for module in [self.mlp, self.user_embedding]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        # self.mlp[-1].bias.data=torch.Tensor([valence_mean,arousal_mean]) # valence, arousal 
    def forward(self, 
                cls_embeddings: torch.Tensor, 
                user_indices: torch.Tensor, 
                is_word_indices: torch.Tensor
    ):
        user_matrix = self.user_embedding(user_indices)
        full_feature_matrix = torch.cat([
            cls_embeddings, 
            user_matrix, 
            is_word_indices
        ], dim=1)
        # full_feature_matrix = self.norm(full_feature_matrix)
        results = self.mlp(full_feature_matrix)
        valence = results[:,0]
        arousal = results[:,1]
        return {
            "arousal": arousal,
            "valence": valence
        }

class VersionDAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4# 1 is for is_words  
        # self.norm = torch.nn.LayerNorm(self.input_dimension_size) 
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
        for module in [self.shared, self.user_embedding]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    def forward(self, 
                cls_embeddings: torch.Tensor, 
                user_indices: torch.Tensor, 
                is_word_indices: torch.Tensor
    ):
        user_matrix = self.user_embedding(user_indices)
        full_feature_matrix = torch.cat([
            cls_embeddings, 
            user_matrix, 
            is_word_indices
        ], dim=1)
        # full_feature_matrix = self.norm(full_feature_matrix)
        shared = self.shared(full_feature_matrix)
        arousal_classification = self.arousal_layer(shared)
        valence_classification = self.valence_layer(shared)
        return {
            "arousal": arousal_classification, # these are logits 
            "valence": valence_classification
        }

class VersionGAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4# 1 is for is_words  
        # self.norm = torch.nn.LayerNorm(self.input_dimension_size) 
        self.hidden_dim = self.input_dimension_size//2
        
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), # CLS embeddings to hidden layer 
            torch.nn.GELU(), # apply a GELU to hidden (read that this works well with RoBerta)
            torch.nn.Dropout(0.1), # 30% of the hidden layer nodes are dropped 
        )
        self.arousal_layer = torch.nn.Linear(self.hidden_dim, 2) 
        self.valence_layer = torch.nn.Linear(self.hidden_dim, 4) 
        self.__initialize_weights__()
    def __initialize_weights__(self):
        for module in [self.shared, self.user_embedding]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    def forward(self, 
                cls_embeddings: torch.Tensor, 
                user_indices: torch.Tensor, 
                is_word_indices: torch.Tensor
    ):
        user_matrix = self.user_embedding(user_indices)
        full_feature_matrix = torch.cat([
            cls_embeddings, 
            user_matrix, 
            is_word_indices
        ], dim=1)
        # full_feature_matrix = self.norm(full_feature_matrix)
        
        hidden_output = self.shared(full_feature_matrix) # currently using sigmoid 
        
        valence_logits = self.valence_layer(hidden_output)
        arousal_logits = self.arousal_layer(hidden_output)
        
        return {
            "valence": valence_logits,
            "arousal": arousal_logits 
        }


class DualAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4 # 1 is for is_words  
        self.norm = torch.nn.LayerNorm(self.input_dimension_size) 
        
        self.hidden_dim = self.input_dimension_size // 2

        self.valence_head = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_dim, 1)
        )

        self.arousal_head = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_dim, 1)
        )
    def __initialize_weights__(self):
        for module in [self.valence_head, self.arousal_head, self.user_embedding]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, cls_embeddings: torch.Tensor, user_indices: torch.Tensor, is_word_indices: torch.Tensor):
        user_matrix = self.user_embedding(user_indices)
        full_feature_matrix = torch.cat([cls_embeddings, user_matrix, is_word_indices.unsqueeze(1)], dim=1)
        full_feature_matrix = self.norm(full_feature_matrix)
        v = torch.sigmoid(self.valence_head(full_feature_matrix))  # [0,1]
        a = torch.sigmoid(self.arousal_head(full_feature_matrix))
        return v.squeeze(-1), a.squeeze(-1)
