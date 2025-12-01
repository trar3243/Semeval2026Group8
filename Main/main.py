#!../bin/python3
import sys, os, random
import torch
import torch.nn.functional as F
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall

SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)

from ClassDefinition.Utils import Logger, ArgumentParser
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta
from ClassDefinition.Dataset import Dataset, Batch
from ClassDefinition.AffectClassifier import AffectClassifier, DualAffectClassifier 
from losses import build_criterion, compute_single_task_loss

required_arguments = []
optional_arguments = {
    "dataPath": f"{SEMROOT}/Data/TRAIN_RELEASE_3SEP2025/train_subtask1.csv",
    "numEpochs": 5,
    "batchSize": 16,
    "learningRate": 1e-3,
}
g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

USAGE = """
main.py
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    g_ArgParse.printArguments()


def trainingLoop(
    model,
    optimizer,
    dataset: Dataset,
):
    """
    Step 7:
      - iterate over train batches
      - forward -> logits
      - compute loss
      - backward, gradient clip, optimizer step
      - log average train loss per epoch
    """
    # Note: we keep the existing pattern of constructing batches and
    # letting Batch.getFeatures() internally use Roberta.
    batch_size = int(g_ArgParse.get("batchSize"))
    num_epochs = int(g_ArgParse.get("numEpochs"))

    dataset.setTrainBatchList(batch_size)
    train_batch_list = dataset.getTrainBatchList()

    criterion = torch.nn.CrossEntropyLoss() # build_criterion()
    number_of_batches = len(train_batch_list)
    print("Beginning training loop")
        
    for epoch in range(num_epochs):
        dataset.shuffle()
        train_losses = []
        evaluate_arousal_mae(model, dataset)
        model.train()
        dataset.roberta.getModel().train()

        for j in range(number_of_batches):
            if j % 10 == 0:
                print(
                    f"\tEpoch:{epoch+1}/{num_epochs}\tbatch:{j}/{number_of_batches}"
                )

            batch = train_batch_list[j]

            # all CLS embeddings of each entry in batch
            cls_embeddings = batch.getClsEmbeddings()  # [B, 768]
            user_indices = batch.getUserIndices()
            is_words = batch.getIsWords() 

            # the labels for for the batch
            arousalLabels = batch.arousalLabelList  # [B] long
            valenceLabels = batch.valenceLabelList  # [B] long
            labels = torch.stack([valenceLabels, arousalLabels], dim=1)

            optimizer.zero_grad()

            # get predictions of the model
            predictions = model(cls_embeddings, user_indices, is_words)  # [B, 2]
                
            valence_logits = predictions["valence_logits"]
            arousal_logits = predictions["arousal_logits"]
            
            valence_loss = criterion(valence_logits, valenceLabels)
            arousal_loss = criterion(arousal_logits, arousalLabels)
            # update training loop to handle two heads separately
            # valence_prediction, arousal_prediction = model(cls_embeddings, user_indices, is_words)

            # valence_loss = criterion(valence_prediction, valenceLabels)
            # arousal_loss = criterion(arousal_prediction, arousalLabels)

            # get loss
            #loss, log = compute_single_task_loss(logits, arousalLabels, criterion)
            loss = valence_loss+arousal_loss
            # backprop loss
            loss.backward()

            # TODO could implement gradient clipping here (small, but helps keep training stable)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.roberta.getParameters(), 1.0)

            # optimizer applies to model
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        print(f"Epoch {epoch+1}/{num_epochs} average train loss: {avg_train_loss:.4f}")


def evaluate_arousal_mae(model: torch.nn.Module, dataset: Dataset) -> float:
    """
    Step 8:
      - Evaluate on the dev set
      - For each dev batch:
          - features -> predictions
      - Compute MAE vs the original float arousal labels
    """
    model.eval()
    dataset.roberta.getModel().eval()
    batch_size = int(g_ArgParse.get("batchSize"))

    dataset.setDevBatchList(batch_size)
    arousal_predictions = []
    valence_predictions = []
    arousal_labels=[]
    valence_labels=[]
    with torch.no_grad():
        for dev_batch in dataset.getDevBatchList():
            cls_embeddings = dev_batch.getClsEmbeddings()
            user_indices = dev_batch.getUserIndices()
            is_words = dev_batch.getIsWords()
            
            logits = model(cls_embeddings, user_indices, is_words) # [B, 2]
            valence_logits = logits["valence_logits"]
            arousal_logits = logits["arousal_logits"]

            valence_predictions_binned = valence_logits.argmax(dim=-1)
            arousal_predictions_binned = arousal_logits.argmax(dim=-1)
            arousal_predictions.append(arousal_predictions_binned)
            valence_predictions.append(valence_predictions_binned)

            valence_labels.append(dev_batch.valenceLabelList)
            arousal_labels.append(dev_batch.arousalLabelList)
            # val_pred, aro_pred = model(cls_embeddings, user_indices, is_words) # [B, 2]

    
    arousal_predictions = torch.cat(arousal_predictions, dim=0)
    valence_predictions = torch.cat(valence_predictions, dim=0)
    arousal_labels = torch.cat(arousal_labels, dim=0)
    valence_labels = torch.cat(valence_labels, dim=0)

    n = max(1, len(arousal_predictions) // 10)
    print(f"Valence predictions: {valence_predictions[:n]}")
    print(f"Valence labels     : {valence_labels[:n]}")
    print(f"Arousal predictions: {arousal_predictions[:n]}")
    print(f"Arousal labels     : {arousal_labels[:n]}")

    valence_mae = (valence_predictions.float() - valence_labels.float()).abs().mean()
    arousal_mae = (arousal_predictions.float() - arousal_labels.float()).abs().mean()
    print(f"Dev MAE — Valence: {valence_mae:.4f}, Arousal: {arousal_mae:.4f}")

    #Separate F1 for Arousal, Valence
    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousal = f1(target=arousal_labels.long(), preds=arousal_predictions.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValence = f1(target=valence_labels.long(), preds=valence_predictions.long())
    print(f"Dev F1 (arousal): {f1ScoreArousal:.4f}")
    print(f"Dev F1 (valence): {f1ScoreValence:.4f}")

    #Separate Accuracy for Arousal, Valence
    AccuracyArousal = Accuracy(task='multiclass', num_classes=3, average='macro')(arousal_predictions.long(), arousal_labels.long())
    AccuracyValence = Accuracy(task='multiclass', num_classes=5, average='macro')(valence_predictions.long(), valence_labels.long())
    print(f"Dev Accuracy (arousal): {AccuracyArousal:.4f}")
    print(f"Dev Accuracy (valence): {AccuracyValence:.4f}")

    #Separate Precision for Arousal, Valence
    PrecisionArousal = Precision(task='multiclass', num_classes=3, average='macro')(arousal_predictions.long(), arousal_labels.long())
    PrecisionValence = Precision(task='multiclass', num_classes=5, average='macro')(valence_predictions.long(), valence_labels.long())
    print(f"Dev Precision (arousal): {PrecisionArousal:.4f}")
    print(f"Dev Precision (valence): {PrecisionValence:.4f}")

    #Separate Recall for Arousal, Valence
    RecallArousal  = Recall(task='multiclass', num_classes=3, average='macro')(arousal_predictions.long(), arousal_labels.long())
    RecallValence  = Recall(task='multiclass', num_classes=5, average='macro')(valence_predictions.long(), valence_labels.long())
    print(f"Dev Recall (arousal): {RecallArousal:.4f}")
    print(f"Dev Recall (valence): {RecallArousal:.4f}")

    #Combined F1 for the 15 class combinations over Arousal, Valence
    #Multiply Arousal (0.0, 1.0, 2.0) to distribute all possible combinations across unique classes (0 - 14)
    Combined_Predictions = (arousal_predictions.long() * 5) + valence_predictions.long()
    Combined_Labels = (arousal_labels.long() * 5) + valence_labels.long() 
    #Calculate F1 from combined labels, predictions
    CombinedF1  = F1Score(task='multiclass', num_classes=15, average='macro')(Combined_Predictions, Combined_Labels)
    print(f"Combined F1: {CombinedF1:.4f}")
    
    return (valence_mae, arousal_mae, f1ScoreArousal, f1ScoreValence, AccuracyArousal, AccuracyValence, PrecisionArousal, PrecisionValence, RecallArousal, RecallValence, CombinedF1)

def save_model_and_bins(model: torch.nn.Module):
    """
    Step 9:
      - Save model weights and the bin centers/edges used for arousal.
    """
    artifacts_dir = os.path.join(SEMROOT, "Artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    save_path = os.path.join(artifacts_dir, "arousal_head.pt")

    # Simple metadata for downstream use.
    bin_centers = [0.0, 1.0, 2.0]
    # "Edges" here are just illustrative; update if you later decide on strict bin edges.
    bin_edges = [0.0, 1.0, 2.0]

    state = {
        "state_dict": model.state_dict(),
        "feature_dim": getattr(model, "input_dimension_size", 768),
        "num_classes": 3,
        "task": "arousal",
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "label_mapping": {0: 0.0, 1: 1.0, 2: 2.0},
    }

    torch.save(state, save_path)
    print(f"Saved model and bin metadata to: {save_path}")


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
    roberta = Roberta()            # create only once
    dataset = Dataset(g_ArgParse.get("dataPath"), roberta)
    dataset.printSetDistribution()
    # 3. Define bins for classification labels:
    #    - Valence: 5 bins over [-2, 2] -> class ids {0..4}
    #    - Arousal: 3 bins over [ 0, 2] -> class ids {0..2}
    
    # 4. Preprocess -> tokenize
    #   - use Hugging Face tokenizer for RoBERTa
    #   - create dataset/dataloader and return input_ids, attention_mask, y_valence_class, y_arousal_class
    
    # 5. Build model
    # model = DualAffectClassifier(dataset.number_of_users) # dual-head model version
    model = AffectClassifier(dataset.number_of_users)

    # 6. Loss and optimizer
    learning_rate = float(g_ArgParse.get("learningRate"))
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": learning_rate},      # head
        {"params": roberta.getParameters(), "lr": 2e-5},    # only last layers
    ])
    
    # 7. Training loop
    trainingLoop(model, optimizer, dataset)

    # 8. Evaluation – compute MAE on dev set
    evaluate_arousal_mae(model, dataset)

    # 9. Save model weights and bin edges/centers
    save_model_and_bins(model)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
