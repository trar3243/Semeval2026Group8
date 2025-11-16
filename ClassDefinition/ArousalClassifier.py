import sys, os
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)
from ClassDefinition.Roberta import Roberta

class ArousalClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dimension_size = Roberta.output_dimension_size 
        # 3 class output 
        self.output_dimension_size = 3 
        # set the coefficients 
        self.coefficients = torch.nn.Linear(self.input_dimension_size, self.output_dimension_size)
        self.__initialize_weights__()
    def __initialize_weights__(self):
        self.coefficients.weight.data.fill_(1.0)# for now, set all to 1.0 

    def forward(self, features: torch.Tensor):
        # We predict a number by multipling by the coefficients
        # then run a softmax over the coefficients 
        #return torch.softmax(self.coefficients(features), dim=1) # dim = 1 
        return self.coefficients(features)
