from transformers import RobertaModel, RobertaTokenizer 
import torch
import logging 
logging.getLogger("transformers").setLevel(logging.ERROR) # Roberta gives warning when not using pooler weights. We are not pooling, so currently disable warning. Revisit if wish to backprop to Roberta. 


"""
Should only be created once per run of main 
private data members: 
    __tokenizer
    __model
public data members: 
    tokens 
    cls_embedding
    text 
"""
class Roberta:
    output_dimension_size = 768 # static data member  
    def __init__(self):
        self.__tokenizer = None 
        self.__model = None
        self.text = None 
        self.tokens = None 
        self.cls_embedding = None 
        self.__init_model()
    def __init_model(self):
        model_name = "roberta-base"  
        self.__tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.__model = RobertaModel.from_pretrained(model_name) 
        # for name, param in self.__model.named_parameters():
        #    param.requires_grad = False
        #    if "layer.11" in name or "layer.10" in name:   # Only train the last two layers 
        #        param.requires_grad = True
    def getTokenizer(self):
        return self.__tokenizer
    def getModel(self):
        return self.__model
    def getParameters(self):
        return self.__model.parameters()

    def setTextList(self, text): # text is str list 
        self.text = text
        self.__tokenizeText()
        self.__updateClsEmbedding()
    def getText(self):
        return self.text 
    def __tokenizeText(self):
        self.tokens=self.__tokenizer(self.text, return_tensors="pt", padding=True, truncation=True) # gotta check what the pt means; found in docs 
        
    
    def __updateClsEmbedding(self):
        if(self.tokens == None):
            raise Exception(f"Fatal error: attempt to update cls embedding without self.tokens set")
        outputs = self.__model(**self.tokens)
        self.cls_embedding = outputs.last_hidden_state[:, 0, :] # last hidden state 
    def getClsEmbedding(self):
        return self.cls_embedding

        
