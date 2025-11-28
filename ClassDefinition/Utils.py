import logging 
import torch 

class Logger:
    def __init__(self, name, level=20): # 10 for debug, 20 for info, 30 for warning, 40 for error, 50 for critical 
        self.logger= logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    def print(self, message, level=20):
        self.logger.log(level, message)

logger = Logger(__name__)
print = logger.print

class ArgumentParser:
    def __init__(self):
        self.arguments = None
    """
    setArguments
    sets self.arguments to be the inputArguments provided, assuming inputArguments is a list of = seperated key, value pairs 
    if an argument key is not in required or optional arguments, it will be ignored 
    """
    def setArguments(self, inputArguments: list, required_arguments:list=[], optional_arguments:dict={}):
        "inputArguments is just what typically gets passed to argc, an array"
        self.arguments={}
        for inputArgument in inputArguments:
            if(not inputArgument.count("=")):
                raise Exception(f"Argument passed to setArguments is not = character set. We assume <variable>=<value> input to the script. Failing argument: {inputArgument}")
            (key,value) = (inputArgument.split('=')[0], inputArgument.split('=')[1])    
            if(key not in required_arguments + list(optional_arguments.keys())):
                print(f"Provided argument {key} not in required or optional arguments. Ignoring.")
            else:
                self.set(key, value)
                
        for required_argument in required_arguments:
            if(required_argument not in self.arguments):
                raise Exception(f"Missing {required_argument} from provided inputs")
        for optional_argument in optional_arguments:
            if(optional_argument not in self.arguments):
                print(f"Optional argument {optional_argument} not found in input. Continuing...")
                self.set(optional_argument, optional_arguments[optional_argument])
    
    def get(self, name):
        if(name not in self.arguments):
            raise Exception(f"Cannot find {name} in arguments.")
        return self.arguments[name]
    def set(self,key,value):
        self.arguments[key]=value 
    def printArguments(self):
        for argument in self.arguments:
            print(f"\tKey:{argument}\tValue:{self.arguments[argument]}")

def convertStringToFloat(string: str): 
    string = string.lower()
    if(string == "false"):
        return 0.0
    elif(string == "true"):
        return 1.0
    else:
        raise Exception(f"Input {string} not in: false, true")

def createSoftLabels(targets, number_of_classes, sigma=0.5): # will return an array with decreasing values away from true label  
    # because the classes are bins, lessen loss for "Close" guesses 
    # the labels sum to 1 
    device = targets.device 
    batch_size = targets.size(0)
    soft_labels = torch.zeros(batch_size, number_of_classes, device=device)

    # Create Gaussian probabilities for each class
    class_indices = torch.arange(number_of_classes, device=device).float().unsqueeze(0)  # (1, num_classes)
    targets_float = targets.float().unsqueeze(1)  # (batch_size, 1)

    # Gaussian probability: exp(-(x - mu)^2 / (2*sigma^2))
    soft_labels = torch.exp(-0.5 * ((class_indices - targets_float) / sigma)**2)

    # Normalize to sum to 1
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
    return soft_labels

    device = targets.device 

    batch_size = targets.size(0) # get number of batches 
    soft_labels = torch.zeros(batch_size, number_of_classes, device=device)

    class_indices = torch.arange(number_of_classes, device=device).float().unsqueeze(0)  # (1, num_classes)
    targets_float = targets.float().unsqueeze(1)  # (batch_size, 1)
    
    scale = 4 / (number_of_classes - 1)
    soft_labels = torch.exp(-torch.abs(class_indices - targets_float) * scale)
    
    # Normalize to sum to 1
    # soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
    return soft_labels
