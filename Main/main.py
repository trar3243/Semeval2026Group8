#!../bin/python3
import sys, os, random
import torch 
SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)

from ClassDefinition.Utils import Logger, ArgumentParser
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta 
from ClassDefinition.Dataset import Dataset 
from ClassDefinition.ArousalClassifier import ArousalClassifier
from losses import build_criterion, compute_single_task_loss

required_arguments=[]
optional_arguments={
    "dataPath":f"{SEMROOT}/Data/TRAIN_RELEASE_3SEP2025/train_subtask1.csv",
    "numEpochs":5, # 60?
    "batchSize":16,
    "learningRate":0.05
}
g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print=g_Logger.print

USAGE="""
main.py
"""

def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    g_ArgParse.printArguments()

def trainingLoop(
    model,
    optimizer,
    dataset
):
    roberta=Roberta()
    dataset.shuffle()
    dataset.setTrainBatchList(g_ArgParse.get("batchSize"))
    criterion = build_criterion()
    number_of_batches = len(dataset.getTrainBatchList())
    print("Beginning training loop")
    for i in range(0, g_ArgParse.get("numEpochs")):
        train_losses=[]
        dev_losses=[]
        for j in range(0, number_of_batches):
            if(j % 10 == 0):
                print(f"\tEpoch:{i}/{g_ArgParse.get('numEpochs')}\tbatch:{j}/{number_of_batches}")
            # all CLS embeddings of each entry in batch 
            features = dataset.getTrainBatchList()[j].getFeatures()
            # the labels for arousals for the batch 
            arousalLabels = dataset.getTrainBatchList()[j].arousalLabelList

            optimizer.zero_grad()

            # get logits of the model (should output 3-way)
            logits = model(features)

            # get loss 
            loss, log = compute_single_task_loss(logits, arousalLabels, criterion) 

            #backprop loss 
            loss.backward()
            
            # optimizer applies to model 
            optimizer.step()

            

def main(inputArguments):
    initialize(inputArguments)
    """
    run Valence OR Arousal in separate runs, controlled by an input tag:
      - task=valence  -> 5-class classification over [-2, 2] bins
      - task=arousal  -> 3-class classification over [0, 2] bins
    """

    # 1. Parse args and set variables (task, model_name, batch_size, epochs, lr, max_length, etc.)
    #   ex. python main.py task=valence max_length=256 batch_size=16 epochs=4 (optional: lr_enc=2e-5 lr_head=1e-3)



    # 2. Load training CSV data (train_subtask1.csv)
    #   - required columns: user_id, text, valence (float in [-2,2]), arousal (float in [0,2])
    #   - drop NaNs, basic whitespace cleanup
    from data_ingest import ingest
    entries = ingest(g_ArgParse.get("dataPath"))
    print(f"{len(entries)} entries ingested")
    # 3. Define bins for classification labels:
    #    - Valence: 5 bins over [-2, 2] -> class ids {0..4}
    #    - Arousal: 3 bins over [ 0, 2] -> class ids {0..2}
    for e in entries:
        #Compute class ids for valence and arousal and update the Entry object.
        e.valence_class = e.valence + 2 #Range 0,1,2,3,4
        e.arousal_class = e.arousal #Range 0,1,2
    # 4. Preprocess -> tokenize
    #   - use Hugging Face tokenzier for RoBERTa (padding=True, truncation=True, max_length=128-256)
    #   - create dataset/dataloader and return input_ids, attention_mask, y_valence_class, y_arousal_class
    dataset = Dataset(entries) # splits into training and dev set 

    # 5. Build model
    #   - Encoder: RoBERTa
    #   - build one head for the chosen task


    # 6. Loss and optimizer
    model = ArousalClassifier()
    optimizer = torch.optim.SGD(model.parameters(), g_ArgParse.get("learningRate"))

    # 7. Training loop
    #   - for each batch:
    #       - forward -> logits [B, K]
    #       - loss, logs = compute_single_task_loss(logits, y_class, criterion)
    #       - backward, clip, optimizer.step(), scheduler.step()
    #       - log loss
    trainingLoop(model,optimizer,dataset)

    # 8. Evaluation
    #    - probs = softmax(logits, dim=-1)
    #    - y_hat_float = sum(probs * bin_centers)  # expected value
    #    - compute MAE vs the float labels (valence or arousal)


    # 9. Save model weights and the bin edges/centers




if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
