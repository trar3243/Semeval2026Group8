import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta

# Version A: Sigmoid regression style (valence, arousal in [0,1])
class VersionAAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) # add +1 to num users for unknown user
        self.user_embedding.weight.requires_grad = True 
        
        # REMOVE LEXICON → drop +6
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4  # 1 is for is_words  
        
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

        logits = self.mlp(full_feature_matrix)
        probs = torch.sigmoid(logits)
        valence_probs = probs[:,0]
        arousal_probs = probs[:,1]
        return {
            "arousal": arousal_probs,
            "valence": valence_probs
        }


# Version B: Direct regression (no sigmoid in the head)
class VersionBAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) 
        self.user_embedding.weight.requires_grad = True 
        
        # REMOVE LEXICON
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4  # 1 is for is_words  
        
        self.hidden_dim = self.input_dimension_size//2
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), 
            torch.nn.GELU(), 
            torch.nn.Dropout(0.1), 
            torch.nn.Linear(self.hidden_dim, 2) 
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

        results = self.mlp(full_feature_matrix)
        valence = results[:,0]
        arousal = results[:,1]
        return {
            "arousal": arousal,
            "valence": valence
        }


# Version D: Shared trunk, softmax heads (3 & 5 classes)
class VersionDAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4) 
        self.user_embedding.weight.requires_grad = True 
        
        # REMOVE LEXICON
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4
        
        self.hidden_dim = self.input_dimension_size//2
        
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), 
            torch.nn.GELU(), 
            torch.nn.Dropout(0.1), 
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

        shared = self.shared(full_feature_matrix)
        arousal_classification = self.arousal_layer(shared)
        valence_classification = self.valence_layer(shared)
        return {
            "arousal": arousal_classification,
            "valence": valence_classification
        }


# Version G: Ordinal (logits) model – 4 valence, 2 arousal
class VersionGAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4)
        self.user_embedding.weight.requires_grad = True 
        
        # REMOVE LEXICON
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4
        
        self.hidden_dim = self.input_dimension_size//2
        
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim), 
            torch.nn.GELU(), 
            torch.nn.Dropout(0.1), 
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

        hidden_output = self.shared(full_feature_matrix)
        
        valence_logits = self.valence_layer(hidden_output)
        arousal_logits = self.arousal_layer(hidden_output)
        
        return {
            "valence": valence_logits,
            "arousal": arousal_logits 
        }


class DualAffectClassifier(torch.nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users,4)
        self.user_embedding.weight.requires_grad = True 
        
        # Already no lexicon in this one
        self.input_dimension_size = Roberta.output_dimension_size + 1 + 4  
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
    

# Version H: Dual affect / dual head with ordinal logic from G
class VersionHAffectClassifier(torch.nn.Module):
    """
    Version H:
      - Dual-head ordinal classifier
      - Same input + shared trunk structure as Version G
      - Valence: 4 ordinal logits (for 5 classes)
      - Arousal: 2 ordinal logits (for 3 classes)
    """
    def __init__(self, num_users):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, 4)
        self.user_embedding.weight.requires_grad = True

        self.input_dimension_size = Roberta.output_dimension_size + 4 + 1
        self.hidden_dim = self.input_dimension_size // 2

        # dual shared trunks
        self.shared_valence = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )

        self.shared_arousal = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_size, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )

        self.valence_head = torch.nn.Linear(self.hidden_dim, 4)
        self.arousal_head = torch.nn.Linear(self.hidden_dim, 2)

        self.__initialize_weights__()


    def __initialize_weights__(self):
        # remove nonexistent self.shared
        for module in [
            self.shared_valence,
            self.shared_arousal,
            self.valence_head,
            self.arousal_head,
            self.user_embedding,
        ]:
            for layer in module.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        cls_embeddings: torch.Tensor,
        user_indices: torch.Tensor,
        is_word_indices: torch.Tensor,
    ):
        # ensure cls_embeddings is 2D [batch, features]
        if cls_embeddings.dim() == 3:
            cls_embeddings = cls_embeddings.squeeze(1)

        # user embedding
        user_matrix = self.user_embedding(user_indices)

        # ensure is_word_indices is [B,1]
        if is_word_indices.dim() == 1:
            is_word_indices = is_word_indices.unsqueeze(1)
        elif is_word_indices.dim() == 3:
            is_word_indices = is_word_indices.squeeze(1)

        full_feature_matrix = torch.cat(
            [cls_embeddings, user_matrix, is_word_indices], dim=1
        )

        # dual shared layers
        hidden_valence = self.shared_valence(full_feature_matrix)
        hidden_arousal = self.shared_arousal(full_feature_matrix)

        valence_logits = self.valence_head(hidden_valence)
        arousal_logits = self.arousal_head(hidden_arousal)

        return {"valence": valence_logits, "arousal": arousal_logits}
