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
from ClassDefinition.AffectClassifier import VersionAAffectClassifier, VersionBAffectClassifier, VersionDAffectClassifier,VersionGAffectClassifier, DualAffectClassifier 
from losses import build_criterion, compute_single_task_loss

required_arguments = []
optional_arguments = {
    "dataPath": f"{SEMROOT}/Data/TRAIN_RELEASE_3SEP2025/train_subtask1.csv",
    "lexiconLookupPath": f"{SEMROOT}/Data/Ratings_Warriner_et_al.csv",
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

def getOrdinalLabel(true_class_indices, num_bins):
    batch_size = true_class_indices.size(0)
    K = num_bins - 1  # number of ordinal output units

    thresholds = torch.arange(K, device=true_class_indices.device).unsqueeze(0)  
    indices_expanded = true_class_indices.unsqueeze(1)                           

    ordinal_matrix = (indices_expanded > thresholds).float()
    return ordinal_matrix

def trainingLoop(
    modelA,modelB,modelD,modelG,
    optimizerA,optimizerB,optimizerD,optimizerG,
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

    criterionA = torch.nn.SmoothL1Loss() 
    criterionB = torch.nn.SmoothL1Loss() 
    criterionD = torch.nn.CrossEntropyLoss()
    criterionG = torch.nn.BCEWithLogitsLoss()
    number_of_batches = len(train_batch_list)
    print("Beginning training loop")
        
    for epoch in range(num_epochs):
        dataset.shuffle()
        train_losses = []
        evaluate_arousal_mae(modelA, modelB, modelD, modelG, dataset)
        modelA.train()
        modelB.train()
        modelD.train()
        modelG.train()
        dataset.robertaA.getModel().train()
        dataset.robertaB.getModel().train()
        dataset.robertaD.getModel().train()
        dataset.robertaG.getModel().train()

        for j in range(number_of_batches):
            if j % 10 == 0:
                print(
                    f"\tEpoch:{epoch+1}/{num_epochs}\tbatch:{j}/{number_of_batches}"
                )

            batch = train_batch_list[j]

            # all CLS embeddings of each entry in batch
            cls_embeddingsA = batch.getClsEmbeddingsA()  # [B, 768]
            cls_embeddingsB = batch.getClsEmbeddingsB()  # [B, 768]
            cls_embeddingsD = batch.getClsEmbeddingsD()  # [B, 768]
            cls_embeddingsG = batch.getClsEmbeddingsG()  # [B, 768]
            user_indices = batch.getUserIndices()
            is_words = batch.getIsWords()

            # the labels for for the batch
            arousalLabels = batch.arousalLabelList  # [B] float -1,...,1
            valenceLabels = batch.valenceLabelList  # [B] float -2,...,2
            
            arousalLabelsA = (arousalLabels + 1)/2 # #0,...,1
            valenceLabelsA = (valenceLabels + 2)/4
            
            arousalLabelsB = arousalLabels 
            valenceLabelsB = valenceLabels
            
            arousalLabelsD = (arousalLabels + 1).to(torch.long) # 0,...,2
            valenceLabelsD = (valenceLabels + 2).to(torch.long) # 0,...,4
            
            arousalLabelsG = getOrdinalLabel(arousalLabelsD, 3)
            valenceLabelsG = getOrdinalLabel(valenceLabelsD, 5)


            optimizerA.zero_grad()
            optimizerB.zero_grad()
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            # get predictions of the model
            predictionsA = modelA(cls_embeddingsA, user_indices, is_words)  # [B, 2]
            predictionsB = modelB(cls_embeddingsB, user_indices, is_words)  # [B, 2]
            predictionsD = modelD(cls_embeddingsD, user_indices, is_words)  # [B, 2]
            predictionsG = modelG(cls_embeddingsG, user_indices, is_words)  # [B, 2]
                
            
            valence_loss_A = criterionA(predictionsA["valence"], valenceLabelsA)
            arousal_loss_A = criterionA(predictionsA["arousal"], arousalLabelsA)
            lossA = valence_loss_A + arousal_loss_A
            lossA.backward()

            valence_loss_B = criterionB(predictionsB["valence"], valenceLabelsB)
            arousal_loss_B = criterionB(predictionsB["arousal"], arousalLabelsB)
            lossB = valence_loss_B + arousal_loss_B
            lossB.backward()

            valence_loss_D = criterionD(predictionsD["valence"], valenceLabelsD)
            arousal_loss_D = criterionD(predictionsD["arousal"], arousalLabelsD)
            lossD = valence_loss_D + arousal_loss_D
            lossD.backward()

            valence_loss_G = criterionG(predictionsG["valence"], valenceLabelsG)
            arousal_loss_G = criterionG(predictionsG["arousal"], arousalLabelsG)
            lossG = valence_loss_G + arousal_loss_G
            lossG.backward()

            
            # update training loop to handle two heads separately
            # valence_prediction, arousal_prediction = model(cls_embeddings, user_indices, is_words)

            # valence_loss = criterion(valence_prediction, valenceLabels)
            # arousal_loss = criterion(arousal_prediction, arousalLabels)

            # get loss
            #loss, log = compute_single_task_loss(logits, arousalLabels, criterion)

            # TODO could implement gradient clipping here (small, but helps keep training stable)
            torch.nn.utils.clip_grad_norm_(modelA.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaA.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelB.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaB.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelD.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaD.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelG.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaG.getParameters(), 1.0)

            # optimizer applies to model
            optimizerA.step()
            optimizerB.step()
            optimizerD.step()
            optimizerG.step()

            train_losses.extend([lossA.item(), lossB.item(), lossD.item(), lossG.item()])

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        print(f"Epoch {epoch+1}/{num_epochs} average train loss: {avg_train_loss:.4f}")


def evaluate_arousal_mae(
    modelA: torch.nn.Module, 
    modelB: torch.nn.Module, 
    modelD: torch.nn.Module, 
    modelG: torch.nn.Module, 
    dataset: Dataset
) -> float:
    """
    Step 8:
      - Evaluate on the dev set
      - For each dev batch:
          - features -> predictions
      - Compute MAE vs the original float arousal labels
    """
    modelA.eval()
    modelB.eval()
    modelD.eval()
    modelG.eval()
    dataset.robertaA.getModel().eval()
    dataset.robertaB.getModel().eval()
    dataset.robertaD.getModel().eval()
    dataset.robertaG.getModel().eval()
    batch_size = int(g_ArgParse.get("batchSize"))

    dataset.setDevBatchList(batch_size)
    arousal_predictions = []
    valence_predictions = []
    arousal_predictions_A = []
    valence_predictions_A = []
    arousal_predictions_B = []
    valence_predictions_B = []
    arousal_predictions_D = []
    valence_predictions_D = []
    arousal_predictions_G = []
    valence_predictions_G = []
    arousal_labels=[]
    valence_labels=[]

    arousal_predictions_continuous_A = []
    valence_predictions_continuous_A = []
    arousal_predictions_continuous_B = []
    valence_predictions_continuous_B = []
    arousal_predictions_continuous_D = []
    valence_predictions_continuous_D = []
    arousal_predictions_continuous_G = []
    valence_predictions_continuous_G = []
    arousal_predictions_continuous = []
    valence_predictions_continuous = []
    
    with torch.no_grad():
        for dev_batch in dataset.getDevBatchList():
            cls_embeddingsA = dev_batch.getClsEmbeddingsA()
            cls_embeddingsB = dev_batch.getClsEmbeddingsB()
            cls_embeddingsD = dev_batch.getClsEmbeddingsD()
            cls_embeddingsG = dev_batch.getClsEmbeddingsG()
            user_indices = dev_batch.getUserIndices()
            is_words = dev_batch.getIsWords()
            
            # gets messy here 
            arousalLabels = (dev_batch.arousalLabelList + 1.0).to(torch.long) # [B] long 0,...,2
            valenceLabels = (dev_batch.valenceLabelList + 2.0).to(torch.long) # [B] long 0,...,4
            

            # get predictions of the model
            predictionsA = modelA(cls_embeddingsA, user_indices, is_words)  # [B, 2]
            predictionsB = modelB(cls_embeddingsB, user_indices, is_words) # [B, 2]
            predictionsD = modelD(cls_embeddingsD, user_indices, is_words) # [B, 2]
            predictionsG = modelG(cls_embeddingsG, user_indices, is_words) # [B, 2]
            
            predictionsAArousal = torch.round(predictionsA["arousal"] * 2)
            predictionsAValence = torch.round(predictionsA["valence"] * 4)
            predictionsAArousalCont = predictionsA["arousal"] * 2
            predictionsAValenceCont = predictionsA["valence"] * 4

            predictionsBArousal = torch.round(
                torch.clamp(predictionsB["arousal"] + 1,
                min=0, max=2)
            )
            predictionsBValence = torch.round(
                torch.clamp(predictionsB["valence"] + 2,
                min=0, max=4)
            )
            predictionsBArousalCont = torch.clamp(predictionsB["arousal"] + 1,min=0.0, max=2.0)
            predictionsBValenceCont = torch.clamp(predictionsB["valence"] + 2,min=0.0, max=4.0)

            predictionsDArousal = predictionsD["arousal"].argmax(dim=-1)
            predictionsDValence = predictionsD["valence"].argmax(dim=-1)
            predictionsDArousalProb = torch.softmax(predictionsD["arousal"], dim=-1)
            predictionsDValenceProb = torch.softmax(predictionsD["valence"], dim=-1)
            arousal_classes = torch.arange(3, device=predictionsDArousalProb.device).float()
            valence_classes = torch.arange(5, device=predictionsDArousalProb.device).float()
            predictionsDArousalCont = (predictionsDArousalProb * arousal_classes).sum(dim=-1) # weighted sums 
            predictionsDValenceCont = (predictionsDValenceProb * valence_classes).sum(dim=-1)

            predictionsGArousal = (torch.sigmoid(predictionsG["arousal"]) > 0.5).sum(dim=1)
            predictionsGValence = (torch.sigmoid(predictionsG["valence"]) > 0.5).sum(dim=1)
            predictionsGArousalCont = (torch.sigmoid(predictionsG["arousal"])).sum(dim=1) # sum of the probabilities 
            predictionsGValenceCont = (torch.sigmoid(predictionsG["valence"])).sum(dim=1)
            

            arousal_predictions_A.append(predictionsAArousal.to(torch.long))
            valence_predictions_A.append(predictionsAValence.to(torch.long))
            arousal_predictions_B.append(predictionsBArousal.to(torch.long))
            valence_predictions_B.append(predictionsBValence.to(torch.long))
            arousal_predictions_D.append(predictionsDArousal.to(torch.long))
            valence_predictions_D.append(predictionsDValence.to(torch.long))
            arousal_predictions_G.append(predictionsGArousal.to(torch.long))
            valence_predictions_G.append(predictionsGValence.to(torch.long))
            
            arousal_predictions_continuous_A.append(predictionsAArousalCont.to(torch.float32))
            valence_predictions_continuous_A.append(predictionsAValenceCont.to(torch.float32))
            arousal_predictions_continuous_B.append(predictionsBArousalCont.to(torch.float32))
            valence_predictions_continuous_B.append(predictionsBValenceCont.to(torch.float32))
            arousal_predictions_continuous_D.append(predictionsDArousalCont.to(torch.float32))
            valence_predictions_continuous_D.append(predictionsDValenceCont.to(torch.float32))
            arousal_predictions_continuous_G.append(predictionsGArousalCont.to(torch.float32))
            valence_predictions_continuous_G.append(predictionsGValenceCont.to(torch.float32))

            arousalVotes = torch.stack([
                predictionsAArousal.to(torch.long),
                predictionsBArousal.to(torch.long),
                predictionsDArousal.to(torch.long),
                predictionsGArousal.to(torch.long)], dim=0
            )
            valenceVotes = torch.stack([
                predictionsAValence.to(torch.long),
                predictionsBValence.to(torch.long),
                predictionsDValence.to(torch.long),
                predictionsGValence.to(torch.long)], dim=0
            )
            arousalVotesCont = torch.stack([
                predictionsAArousalCont.to(torch.float32),
                predictionsBArousalCont.to(torch.float32),
                predictionsDArousalCont.to(torch.float32),
                predictionsGArousalCont.to(torch.float32)], dim=0
            )
            valenceVotesCont = torch.stack([
                predictionsAValenceCont.to(torch.float32),
                predictionsBValenceCont.to(torch.float32),
                predictionsDValenceCont.to(torch.float32),
                predictionsGValenceCont.to(torch.float32)], dim=0
            )

            arousal_predicted = torch.mode(arousalVotes, dim=0).values
            valence_predicted = torch.mode(valenceVotes, dim=0).values
            arousal_predicted_cont = torch.mean(arousalVotesCont, dim=0)
            valence_predicted_cont = torch.mean(valenceVotesCont, dim=0)

            arousal_predictions.append(arousal_predicted)
            valence_predictions.append(valence_predicted)
            arousal_predictions_continuous.append(arousal_predicted_cont)
            valence_predictions_continuous.append(valence_predicted_cont)

            valence_labels.append((dev_batch.valenceLabelList + 2.0).to(torch.long))
            arousal_labels.append((dev_batch.arousalLabelList + 1.0).to(torch.long))
            # val_pred, aro_pred = model(cls_embeddings, user_indices, is_words) # [B, 2]

    arousal_predictions = torch.cat(arousal_predictions, dim=0)
    valence_predictions = torch.cat(valence_predictions, dim=0)
    arousal_predictions_A = torch.cat(arousal_predictions_A, dim=0)
    valence_predictions_A = torch.cat(valence_predictions_A, dim=0)
    arousal_predictions_B = torch.cat(arousal_predictions_B, dim=0)
    valence_predictions_B = torch.cat(valence_predictions_B, dim=0)
    arousal_predictions_D = torch.cat(arousal_predictions_D, dim=0)
    valence_predictions_D = torch.cat(valence_predictions_D, dim=0)
    arousal_predictions_G = torch.cat(arousal_predictions_G, dim=0)
    valence_predictions_G = torch.cat(valence_predictions_G, dim=0)
    arousal_labels = torch.cat(arousal_labels, dim=0)
    valence_labels = torch.cat(valence_labels, dim=0)

    valence_predictions_continuous = torch.cat(valence_predictions_continuous, dim=0)
    arousal_predictions_continuous = torch.cat(arousal_predictions_continuous, dim=0)
    valence_predictions_continuous_A = torch.cat(valence_predictions_continuous_A, dim=0)
    arousal_predictions_continuous_A = torch.cat(arousal_predictions_continuous_A, dim=0)
    valence_predictions_continuous_B = torch.cat(valence_predictions_continuous_B, dim=0)
    arousal_predictions_continuous_B = torch.cat(arousal_predictions_continuous_B, dim=0)
    valence_predictions_continuous_D = torch.cat(valence_predictions_continuous_D, dim=0)
    arousal_predictions_continuous_D = torch.cat(arousal_predictions_continuous_D, dim=0)
    valence_predictions_continuous_G = torch.cat(valence_predictions_continuous_G, dim=0)
    arousal_predictions_continuous_G = torch.cat(arousal_predictions_continuous_G, dim=0)
    
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
    #Separate F1 for Arousal, Valence
    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousal = f1(target=arousal_labels.long(), preds=arousal_predictions_A.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValence = f1(target=valence_labels.long(), preds=valence_predictions_A.long())
    print(f"Dev F1 A (arousal): {f1ScoreArousal:.4f}")
    print(f"Dev F1 A (valence): {f1ScoreValence:.4f}")
    #Separate F1 for Arousal, Valence
    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousal = f1(target=arousal_labels.long(), preds=arousal_predictions_B.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValence = f1(target=valence_labels.long(), preds=valence_predictions_B.long())
    print(f"Dev F1 B (arousal): {f1ScoreArousal:.4f}")
    print(f"Dev F1 B (valence): {f1ScoreValence:.4f}")
    #Separate F1 for Arousal, Valence
    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousal = f1(target=arousal_labels.long(), preds=arousal_predictions_D.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValence = f1(target=valence_labels.long(), preds=valence_predictions_D.long())
    print(f"Dev F1 D (arousal): {f1ScoreArousal:.4f}")
    print(f"Dev F1 D (valence): {f1ScoreValence:.4f}")
    #Separate F1 for Arousal, Valence
    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousal = f1(target=arousal_labels.long(), preds=arousal_predictions_G.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValence = f1(target=valence_labels.long(), preds=valence_predictions_G.long())
    print(f"Dev F1 G (arousal): {f1ScoreArousal:.4f}")
    print(f"Dev F1 G (valence): {f1ScoreValence:.4f}")

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

    #Pearson R
    arousal_float_labels = arousal_labels.float().view(-1)
    valence_float_labels = valence_labels.float().view(-1)
    
    arousal_float_predictions = arousal_predictions_continuous.float().view(-1)
    valence_float_predictions = valence_predictions_continuous.float().view(-1)
    pearson_arousal = torch.corrcoef(torch.stack([arousal_float_predictions,arousal_float_labels]))[0,1]
    pearson_valence = torch.corrcoef(torch.stack([valence_float_predictions,valence_float_labels]))[0,1]
    print(f"Pearson R (arousal): {pearson_arousal:.4f}")
    print(f"Pearson R (valence): {pearson_valence:.4f}")
    
    arousal_float_predictions = arousal_predictions_continuous_A.float().view(-1)
    valence_float_predictions = valence_predictions_continuous_A.float().view(-1)
    pearson_arousal = torch.corrcoef(torch.stack([arousal_float_predictions,arousal_float_labels]))[0,1]
    pearson_valence = torch.corrcoef(torch.stack([valence_float_predictions,valence_float_labels]))[0,1]
    print(f"Pearson R (arousal) A: {pearson_arousal:.4f}")
    print(f"Pearson R (valence) A: {pearson_valence:.4f}")
    
    
    arousal_float_predictions = arousal_predictions_continuous_B.float().view(-1)
    valence_float_predictions = valence_predictions_continuous_B.float().view(-1)
    pearson_arousal = torch.corrcoef(torch.stack([arousal_float_predictions,arousal_float_labels]))[0,1]
    pearson_valence = torch.corrcoef(torch.stack([valence_float_predictions,valence_float_labels]))[0,1]
    print(f"Pearson R (arousal) B: {pearson_arousal:.4f}")
    print(f"Pearson R (valence) B: {pearson_valence:.4f}")
    
    
    arousal_float_predictions = arousal_predictions_continuous_D.float().view(-1)
    valence_float_predictions = valence_predictions_continuous_D.float().view(-1)
    pearson_arousal = torch.corrcoef(torch.stack([arousal_float_predictions,arousal_float_labels]))[0,1]
    pearson_valence = torch.corrcoef(torch.stack([valence_float_predictions,valence_float_labels]))[0,1]
    print(f"Pearson R (arousal) D: {pearson_arousal:.4f}")
    print(f"Pearson R (valence) D: {pearson_valence:.4f}")
    
    
    arousal_float_predictions = arousal_predictions_continuous_G.float().view(-1)
    valence_float_predictions = valence_predictions_continuous_G.float().view(-1)
    pearson_arousal = torch.corrcoef(torch.stack([arousal_float_predictions,arousal_float_labels]))[0,1]
    pearson_valence = torch.corrcoef(torch.stack([valence_float_predictions,valence_float_labels]))[0,1]
    print(f"Pearson R (arousal) G: {pearson_arousal:.4f}")
    print(f"Pearson R (valence) G: {pearson_valence:.4f}")
    
    
    return (valence_mae, arousal_mae, f1ScoreArousal, f1ScoreValence, AccuracyArousal, AccuracyValence, PrecisionArousal, PrecisionValence, RecallArousal, RecallValence, CombinedF1, pearson_arousal, pearson_valence)

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
    robertaA = Roberta()            # create only once
    robertaB = Roberta()            # create only once
    robertaD = Roberta()            # create only once
    robertaG = Roberta()            # create only once
    dataset = Dataset(
        g_ArgParse.get("dataPath"), 
        robertaA, robertaB, robertaD, robertaG
    )
    dataset.printSetDistribution()
    # 3. Define bins for classification labels:
    #    - Valence: 5 bins over [-2, 2] -> class ids {0..4}
    #    - Arousal: 3 bins over [ 0, 2] -> class ids {0..2}
    
    # 4. Preprocess -> tokenize
    #   - use Hugging Face tokenizer for RoBERTa
    #   - create dataset/dataloader and return input_ids, attention_mask, y_valence_class, y_arousal_class
    
    # 5. Build model
    # model = DualAffectClassifier(dataset.number_of_users) # dual-head model version
    modelA = VersionAAffectClassifier(dataset.number_of_users)
    modelB = VersionBAffectClassifier(dataset.number_of_users)
    modelD = VersionDAffectClassifier(dataset.number_of_users)
    modelG = VersionGAffectClassifier(dataset.number_of_users)

    # 6. Loss and optimizer
    learning_rate = float(g_ArgParse.get("learningRate"))
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizerA = torch.optim.AdamW([
        {"params": modelA.parameters(), "lr": learning_rate},      # head
        {"params": robertaA.getParameters(), "lr": 2e-5},    # only last layers
    ])
    optimizerB = torch.optim.AdamW([
        {"params": modelB.parameters(), "lr": learning_rate},      # head
        {"params": robertaB.getParameters(), "lr": 2e-5},    # only last layers
    ])
    optimizerD = torch.optim.AdamW([
        {"params": modelD.parameters(), "lr": learning_rate},      # head
        {"params": robertaD.getParameters(), "lr": 2e-5},    # only last layers
    ])
    optimizerG = torch.optim.AdamW([
        {"params": modelG.parameters(), "lr": learning_rate},      # head
        {"params": robertaG.getParameters(), "lr": 2e-5},    # only last layers
    ])
    
    # 7. Training loop
    trainingLoop(
        modelA, modelB, modelD, modelG,
        optimizerA, optimizerB, optimizerD, optimizerG, 
        dataset
    )

    # 8. Evaluation – compute MAE on dev set
    evaluate_arousal_mae(
        modelA, modelB, modelD, modelG, 
        dataset
    )

    # 9. Save model weights and bin edges/centers
    save_model_and_bins(modelA)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
