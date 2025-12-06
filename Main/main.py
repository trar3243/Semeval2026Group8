#!../bin/python3
import sys, os, random
import torch
import torch.nn.functional as F
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 

SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)

from ClassDefinition.Utils import Logger, ArgumentParser
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta
from ClassDefinition.Dataset import Dataset, Batch
from ClassDefinition.AffectClassifier import VersionAAffectClassifier, VersionBAffectClassifier, VersionDAffectClassifier, VersionGAffectClassifier, VersionHAffectClassifier
from losses import build_criterion, compute_single_task_loss

required_arguments = []
optional_arguments = {
    "dataPath": f"{SEMROOT}/Data/TRAIN_RELEASE_3SEP2025/train_subtask1.csv",
    "numEpochs": 5,
    "batchSize": 16,
    "learningRate": 1e-3,
    "modelSaveRoot": f"{SEMROOT}/Artifacts/",
    "modelSaveSpecific": f"{SEMROOT}/Artifacts/epoch=4/models.pt",
    "devSetPath": f"{SEMROOT}/Data/devset.csv",
    "trainSetPath": f"{SEMROOT}/Data/trainset.csv",
    "mode": "eval"
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
    if(g_ArgParse.get("mode") not in ["train", "eval"]):
        raise Exception(f"Supplied mode {g_ArgParse.get('mode')} not in train,eval")
    g_ArgParse.printArguments()

def getOrdinalLabel(true_class_indices, num_bins):
    batch_size = true_class_indices.size(0)
    K = num_bins - 1  # number of ordinal output units

    thresholds = torch.arange(K, device=true_class_indices.device).unsqueeze(0)  
    indices_expanded = true_class_indices.unsqueeze(1)                           

    ordinal_matrix = (indices_expanded > thresholds).float()
    return ordinal_matrix

def trainingLoop(
    modelA,modelB,modelD,modelG,modelH,
    optimizerA,optimizerB,optimizerD,optimizerG,optimizerH,
    dataset: Dataset,
):
    batch_size = int(g_ArgParse.get("batchSize"))
    num_epochs = int(g_ArgParse.get("numEpochs"))

    dataset.setTrainBatchList(batch_size)
    train_batch_list = dataset.getTrainBatchList()

    criterionA = torch.nn.SmoothL1Loss() 
    criterionB = torch.nn.SmoothL1Loss() 
    criterionD = torch.nn.CrossEntropyLoss()
    criterionG = torch.nn.BCEWithLogitsLoss()
    criterionH = torch.nn.BCEWithLogitsLoss()
    number_of_batches = len(train_batch_list)
    evaluate_arousal_mae(modelA, modelB, modelD, modelG, modelH, dataset)
    epoch_save_path = g_ArgParse.get("modelSaveRoot") + f"epoch=0/models.pt" 
    save_model_and_bins(
        modelA, modelB, modelD,modelG,modelH,
        dataset.robertaA, dataset.robertaB, dataset.robertaD, dataset.robertaG, dataset.robertaH,
        epoch_save_path
    )
    print("Beginning training loop")
        
    for epoch in range(num_epochs):
        dataset.shuffle()
        train_losses = []
        modelA.train()
        modelB.train()
        modelD.train()
        modelG.train()
        modelH.train()
        dataset.robertaA.getModel().train()
        dataset.robertaB.getModel().train()
        dataset.robertaD.getModel().train()
        dataset.robertaG.getModel().train()
        dataset.robertaH.getModel().train()

        for j in range(number_of_batches):
            if j % 10 == 0:
                print(
                    f"\tEpoch:{epoch+1}/{num_epochs}\tbatch:{j}/{number_of_batches}"
                )

            batch = train_batch_list[j]

            # CLS embeddings
            cls_embeddingsA = batch.getClsEmbeddingsA()
            cls_embeddingsB = batch.getClsEmbeddingsB()
            cls_embeddingsD = batch.getClsEmbeddingsD()
            cls_embeddingsG = batch.getClsEmbeddingsG()
            cls_embeddingsH = batch.getClsEmbeddingsH()
            user_indices = batch.getUserIndices()
            is_words = batch.getIsWords()

            # labels
            arousalLabels = batch.arousalLabelList
            valenceLabels = batch.valenceLabelList
            
            arousalLabelsA = (arousalLabels + 1)/2
            valenceLabelsA = (valenceLabels + 2)/4
            
            arousalLabelsB = arousalLabels
            valenceLabelsB = valenceLabels
            
            arousalLabelsD = (arousalLabels + 1).to(torch.long)
            valenceLabelsD = (valenceLabels + 2).to(torch.long)
            
            arousalLabelsG = getOrdinalLabel(arousalLabelsD, 3)
            valenceLabelsG = getOrdinalLabel(valenceLabelsD, 5)

            # Version H uses the same ordinal labels as G
            arousalLabelsH = arousalLabelsG
            valenceLabelsH = valenceLabelsG

            optimizerA.zero_grad()
            optimizerB.zero_grad()
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            optimizerH.zero_grad()

            # model predictions (removed lexicon)
            predictionsA = modelA(cls_embeddingsA, user_indices, is_words)
            predictionsB = modelB(cls_embeddingsB, user_indices, is_words)
            predictionsD = modelD(cls_embeddingsD, user_indices, is_words)
            predictionsG = modelG(cls_embeddingsG, user_indices, is_words)
            predictionsH = modelH(cls_embeddingsH, user_indices, is_words)

            # losses
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

            # Version G: ordinal BCE-with-logits
            valence_loss_G = criterionG(predictionsG["valence"], valenceLabelsG)
            arousal_loss_G = criterionG(predictionsG["arousal"], arousalLabelsG)
            lossG = valence_loss_G + arousal_loss_G
            lossG.backward()


            # Version H: dual ordinal
            valence_loss_H = criterionH(predictionsH["valence"], valenceLabelsH)
            arousal_loss_H = criterionH(predictionsH["arousal"], arousalLabelsH)
            lossH = valence_loss_H + arousal_loss_H
            lossH.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(modelA.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaA.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelB.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaB.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelD.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaD.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelG.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaG.getParameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(modelH.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaH.getParameters(), 1.0)

            # optimizers
            optimizerA.step()
            optimizerB.step()
            optimizerD.step()
            optimizerG.step()
            optimizerH.step()

            train_losses.extend([lossA.item(), lossB.item(), lossD.item(), lossG.item(), lossH.item()])

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        print(f"Epoch {epoch+1}/{num_epochs} average train loss: {avg_train_loss:.4f}")
        
        evaluate_arousal_mae(modelA, modelB, modelD, modelG, modelH, dataset)
        
        epoch_save_path = g_ArgParse.get("modelSaveRoot") + f"epoch={epoch+1}/models.pt" 
        save_model_and_bins(
            modelA, modelB, modelD,modelG,modelH,
            dataset.robertaA, dataset.robertaB, dataset.robertaD, dataset.robertaG, dataset.robertaH,
            epoch_save_path
        )


def evaluate_arousal_mae(
    modelA: torch.nn.Module, 
    modelB: torch.nn.Module, 
    modelD: torch.nn.Module, 
    modelG: torch.nn.Module, 
    modelH: torch.nn.Module, 
    dataset: Dataset
) -> float:

    modelA.eval()
    modelB.eval()
    modelD.eval()
    modelG.eval()
    modelH.eval()
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
    arousal_predictions_H = []
    valence_predictions_H = []
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
    arousal_predictions_continuous_H = []
    valence_predictions_continuous_H = []
    arousal_predictions_continuous = []
    valence_predictions_continuous = []
    
    with torch.no_grad():
        for dev_batch in dataset.getDevBatchList():
            cls_embeddingsA = dev_batch.getClsEmbeddingsA()
            cls_embeddingsB = dev_batch.getClsEmbeddingsB()
            cls_embeddingsD = dev_batch.getClsEmbeddingsD()
            cls_embeddingsG = dev_batch.getClsEmbeddingsG()
            cls_embeddingsH = dev_batch.getClsEmbeddingsH()
            user_indices = dev_batch.getUserIndices()
            is_words = dev_batch.getIsWords()
            
            # class labels
            arousalLabels = (dev_batch.arousalLabelList + 1.0).to(torch.long)
            valenceLabels = (dev_batch.valenceLabelList + 2.0).to(torch.long)
            

            # get predictions of the model
            predictionsA = modelA(cls_embeddingsA, user_indices, is_words)
            predictionsB = modelB(cls_embeddingsB, user_indices, is_words)
            predictionsD = modelD(cls_embeddingsD, user_indices, is_words)
            predictionsG = modelG(cls_embeddingsG, user_indices, is_words)
            predictionsH = modelH(cls_embeddingsH, user_indices, is_words)

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
            

            predictionsHArousal = (torch.sigmoid(predictionsH["arousal"]) > 0.5).sum(dim=1)
            predictionsHValence = (torch.sigmoid(predictionsH["valence"]) > 0.5).sum(dim=1)
            predictionsHArousalCont = (torch.sigmoid(predictionsH["arousal"])).sum(dim=1) # sum of the probabilities 
            predictionsHValenceCont = (torch.sigmoid(predictionsH["valence"])).sum(dim=1)

            arousal_predictions_A.append(predictionsAArousal.to(torch.long))
            valence_predictions_A.append(predictionsAValence.to(torch.long))
            arousal_predictions_B.append(predictionsBArousal.to(torch.long))
            valence_predictions_B.append(predictionsBValence.to(torch.long))
            arousal_predictions_D.append(predictionsDArousal.to(torch.long))
            valence_predictions_D.append(predictionsDValence.to(torch.long))
            arousal_predictions_G.append(predictionsGArousal.to(torch.long))
            valence_predictions_G.append(predictionsGValence.to(torch.long))
            arousal_predictions_H.append(predictionsHArousal.to(torch.long))
            valence_predictions_H.append(predictionsHValence.to(torch.long))
            
            arousal_predictions_continuous_A.append(predictionsAArousalCont.to(torch.float32))
            valence_predictions_continuous_A.append(predictionsAValenceCont.to(torch.float32))
            arousal_predictions_continuous_B.append(predictionsBArousalCont.to(torch.float32))
            valence_predictions_continuous_B.append(predictionsBValenceCont.to(torch.float32))
            arousal_predictions_continuous_D.append(predictionsDArousalCont.to(torch.float32))
            valence_predictions_continuous_D.append(predictionsDValenceCont.to(torch.float32))
            arousal_predictions_continuous_G.append(predictionsGArousalCont.to(torch.float32))
            valence_predictions_continuous_G.append(predictionsGValenceCont.to(torch.float32))
            arousal_predictions_continuous_H.append(predictionsHArousalCont.to(torch.float32))
            valence_predictions_continuous_H.append(predictionsHValenceCont.to(torch.float32))

            arousalVotes = torch.stack([
                predictionsAArousal.to(torch.long),
                predictionsBArousal.to(torch.long),
                predictionsDArousal.to(torch.long),
                predictionsGArousal.to(torch.long),
                predictionsHArousal.to(torch.long)], dim=0
            )
            valenceVotes = torch.stack([
                predictionsAValence.to(torch.long),
                predictionsBValence.to(torch.long),
                predictionsDValence.to(torch.long),
                predictionsGValence.to(torch.long),
                predictionsHValence.to(torch.long)], dim=0
            )
            arousalVotesCont = torch.stack([
                predictionsAArousalCont.to(torch.float32),
                predictionsBArousalCont.to(torch.float32),
                predictionsDArousalCont.to(torch.float32),
                predictionsGArousalCont.to(torch.float32),
                predictionsHArousalCont.to(torch.float32)], dim=0
            )
            valenceVotesCont = torch.stack([
                predictionsAValenceCont.to(torch.float32),
                predictionsBValenceCont.to(torch.float32),
                predictionsDValenceCont.to(torch.float32),
                predictionsGValenceCont.to(torch.float32),
                predictionsHValenceCont.to(torch.float32)], dim=0
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
    arousal_predictions_H = torch.cat(arousal_predictions_H, dim=0)
    valence_predictions_H = torch.cat(valence_predictions_H, dim=0)
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
    valence_predictions_continuous_H = torch.cat(valence_predictions_continuous_H, dim=0)
    arousal_predictions_continuous_H = torch.cat(arousal_predictions_continuous_H, dim=0)
    
    n = max(1, len(arousal_predictions) // 10)
    print(f"Valence predictions: {valence_predictions[:n]}")
    print(f"Valence labels     : {valence_labels[:n]}")
    print(f"Arousal predictions: {arousal_predictions[:n]}")
    print(f"Arousal labels     : {arousal_labels[:n]}")

    valence_mae = (valence_predictions.float() - valence_labels.float()).abs().mean()
    arousal_mae = (arousal_predictions.float() - arousal_labels.float()).abs().mean()
    print(f"Dev MAE — Valence: {valence_mae:.4f}, Arousal: {arousal_mae:.4f}")

    # F1, accuracy, precision, recall, combined evaluation metrics
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
    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousalH = f1(target=arousal_labels.long(), preds=arousal_predictions_H.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValenceH = f1(target=valence_labels.long(), preds=valence_predictions_H.long())
    print(f"Dev F1 H (arousal): {f1ScoreArousalH:.4f}")
    print(f"Dev F1 H (valence): {f1ScoreValenceH:.4f}")

    AccuracyArousal = Accuracy(task='multiclass', num_classes=3, average='macro')(arousal_predictions.long(), arousal_labels.long())
    AccuracyValence = Accuracy(task='multiclass', num_classes=5, average='macro')(valence_predictions.long(), valence_labels.long())
    print(f"Dev Accuracy (arousal): {AccuracyArousal:.4f}")
    print(f"Dev Accuracy (valence): {AccuracyValence:.4f}")

    PrecisionArousal = Precision(task='multiclass', num_classes=3, average='macro')(arousal_predictions.long(), arousal_labels.long())
    PrecisionValence = Precision(task='multiclass', num_classes=5, average='macro')(valence_predictions.long(), valence_labels.long())
    print(f"Dev Precision (arousal): {PrecisionArousal:.4f}")
    print(f"Dev Precision (valence): {PrecisionValence:.4f}")

    RecallArousal  = Recall(task='multiclass', num_classes=3, average='macro')(arousal_predictions.long(), arousal_labels.long())
    RecallValence  = Recall(task='multiclass', num_classes=5, average='macro')(valence_predictions.long(), valence_labels.long())
    print(f"Dev Recall (arousal): {RecallArousal:.4f}")
    print(f"Dev Recall (valence): {RecallValence:.4f}")

    Combined_Predictions = (arousal_predictions.long() * 5) + valence_predictions.long()
    Combined_Labels = (arousal_labels.long() * 5) + valence_labels.long() 
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
    
    arousal_float_predictions = arousal_predictions_continuous_H.float().view(-1)
    valence_float_predictions = valence_predictions_continuous_H.float().view(-1)
    pearson_arousal = torch.corrcoef(torch.stack([arousal_float_predictions,arousal_float_labels]))[0,1]
    pearson_valence = torch.corrcoef(torch.stack([valence_float_predictions,valence_float_labels]))[0,1]
    print(f"Pearson R (arousal) H: {pearson_arousal:.4f}")
    print(f"Pearson R (valence) H: {pearson_valence:.4f}")
    
    #SEEN VS UNSEEN PEARSON R
    all_user_indices = torch.cat([batch.getUserIndices() for batch in dataset.getDevBatchList()], dim=0)
    #Seen mask and unseen mask
    seen_mask = torch.tensor([uid in dataset.train_user_ids for uid in all_user_indices], dtype = torch.bool)
    unseen_mask = torch.tensor([uid not in dataset.train_user_ids for uid in all_user_indices], dtype = torch.bool)

    #Seen User Pearson
    if seen_mask.sum() > 0:
        pearson_arousal_seen = torch.corrcoef(torch.stack([arousal_predictions_continuous[seen_mask].float(), arousal_labels[seen_mask].float()]))[0,1]
        pearson_valence_seen = torch.corrcoef(torch.stack([valence_predictions_continuous[seen_mask].float(), valence_labels[seen_mask].float()]))[0,1]

        print(f"Pearson R (arousal) Seen Users: {pearson_arousal_seen:.4f}")
        print(f"Pearson R (valence) Seen Users: {pearson_valence_seen:.4f}")
    else:
        pearson_arousal_seen = float('nan')
        pearson_valence_seen = float('nan')

    #Unseen User Pearson
    if unseen_mask.sum() > 0:
        pearson_arousal_unseen = torch.corrcoef(torch.stack([arousal_predictions_continuous[unseen_mask].float(), arousal_labels[unseen_mask].float()]))[0,1]
        pearson_valence_unseen = torch.corrcoef(torch.stack([valence_predictions_continuous[unseen_mask].float(), valence_labels[unseen_mask].float()]))[0,1]

        print(f"Pearson R (arousal) Unseen Users: {pearson_arousal_unseen:.4f}")
        print(f"Pearson R (valence) Unseen Users: {pearson_valence_unseen:.4f}")
    else:
        pearson_arousal_unseen = float('nan')
        pearson_valence_unseen = float('nan')
    
    #CONFUSION MATRICIES
    arousal_predictions_numpy = arousal_predictions.cpu().numpy()
    arousal_labels_numpy = arousal_labels.cpu().numpy()
    valence_predictions_numpy = valence_predictions.cpu().numpy()
    valence_labels_numpy = valence_labels.cpu().numpy()

    arousal_confusion = confusion_matrix(arousal_labels_numpy,arousal_predictions_numpy)
    valence_confusion = confusion_matrix(valence_labels_numpy,valence_predictions_numpy)

    #SEABORN CONFUSION MATRIX PLOTS
    arousal_axes = [-1, 0, 1]
    valence_axes = [-2, -1, 0, 1, 2]
    
    plt.figure(figsize=(6,5))
    sns.heatmap(arousal_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=arousal_axes, yticklabels=arousal_axes)
    plt.title("Composite - Arousal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(valence_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=valence_axes, yticklabels=valence_axes)
    plt.title("Composite - Valence Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    #INDIVIDUAL MODEL MATRICIES
    arousal_predictions_numpy_A = arousal_predictions_A.cpu().numpy()
    valence_predictions_numpy_A = valence_predictions_A.cpu().numpy()
    arousal_predictions_numpy_B = arousal_predictions_B.cpu().numpy()
    valence_predictions_numpy_B = valence_predictions_B.cpu().numpy()
    arousal_predictions_numpy_D = arousal_predictions_D.cpu().numpy()
    valence_predictions_numpy_D = valence_predictions_D.cpu().numpy()
    arousal_predictions_numpy_G = arousal_predictions_G.cpu().numpy()
    valence_predictions_numpy_G = valence_predictions_G.cpu().numpy()
    arousal_predictions_numpy_H = arousal_predictions_H.cpu().numpy()
    valence_predictions_numpy_H = valence_predictions_H.cpu().numpy()

    arousal_confusion_A = confusion_matrix(arousal_labels_numpy,arousal_predictions_numpy_A)
    valence_confusion_A = confusion_matrix(valence_labels_numpy,valence_predictions_numpy_A)
    arousal_confusion_B = confusion_matrix(arousal_labels_numpy,arousal_predictions_numpy_B)
    valence_confusion_B = confusion_matrix(valence_labels_numpy,valence_predictions_numpy_B)
    arousal_confusion_D = confusion_matrix(arousal_labels_numpy,arousal_predictions_numpy_D)
    valence_confusion_D = confusion_matrix(valence_labels_numpy,valence_predictions_numpy_D)
    arousal_confusion_G = confusion_matrix(arousal_labels_numpy,arousal_predictions_numpy_G)
    valence_confusion_G = confusion_matrix(valence_labels_numpy,valence_predictions_numpy_G)
    arousal_confusion_H = confusion_matrix(arousal_labels_numpy,arousal_predictions_numpy_H)
    valence_confusion_H = confusion_matrix(valence_labels_numpy,valence_predictions_numpy_H)

    plt.figure(figsize=(6,5))
    sns.heatmap(arousal_confusion_A, annot=True, fmt="d", cmap="Blues", xticklabels=arousal_axes, yticklabels=arousal_axes)
    plt.title("MODEL A - Arousal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(valence_confusion_A, annot=True, fmt="d", cmap="Blues", xticklabels=valence_axes, yticklabels=valence_axes)
    plt.title("MODEL A - Valence Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(arousal_confusion_B, annot=True, fmt="d", cmap="Blues", xticklabels=arousal_axes, yticklabels=arousal_axes)
    plt.title("MODEL B - Arousal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(valence_confusion_B, annot=True, fmt="d", cmap="Blues", xticklabels=valence_axes, yticklabels=valence_axes)
    plt.title("MODEL B - Valence Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(arousal_confusion_D, annot=True, fmt="d", cmap="Blues", xticklabels=arousal_axes, yticklabels=arousal_axes)
    plt.title("MODEL D - Arousal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(valence_confusion_D, annot=True, fmt="d", cmap="Blues", xticklabels=valence_axes, yticklabels=valence_axes)
    plt.title("MODEL D - Valence Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(arousal_confusion_G, annot=True, fmt="d", cmap="Blues", xticklabels=arousal_axes, yticklabels=arousal_axes)
    plt.title("MODEL G - Arousal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(valence_confusion_G, annot=True, fmt="d", cmap="Blues", xticklabels=valence_axes, yticklabels=valence_axes)
    plt.title("MODEL G - Valence Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    plt.figure(figsize=(6,5))
    sns.heatmap(arousal_confusion_H, annot=True, fmt="d", cmap="Blues", xticklabels=arousal_axes, yticklabels=arousal_axes)
    plt.title("MODEL H - Arousal Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6,5))
    sns.heatmap(valence_confusion_H, annot=True, fmt="d", cmap="Blues", xticklabels=valence_axes, yticklabels=valence_axes)
    plt.title("MODEL H - Valence Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


    
    return (valence_mae, arousal_mae, f1ScoreArousal, f1ScoreValence, AccuracyArousal, AccuracyValence, PrecisionArousal, PrecisionValence, RecallArousal, RecallValence, CombinedF1, pearson_arousal, pearson_valence, pearson_arousal_unseen, pearson_valence_unseen, pearson_arousal_seen, pearson_valence_seen)

def save_model_and_bins(
        modelA, modelB, modelD, modelG, modelH, 
        robertaA, robertaB, robertaD, robertaG, robertaH,
        path
    ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "modelA": modelA.state_dict(),
        "modelB": modelB.state_dict(),
        "modelD": modelD.state_dict(),
        "modelG": modelG.state_dict(),
        "modelH": modelH.state_dict(),
        "robertaA": robertaA.getModel().state_dict(),
        "robertaB": robertaB.getModel().state_dict(),
        "robertaD": robertaD.getModel().state_dict(),
        "robertaG": robertaG.getModel().state_dict(),
        "robertaH": robertaH.getModel().state_dict()
    }, path)
    print(f"Saved models to {path}")



def main(inputArguments):
    initialize(inputArguments)
    eval_mode = (g_ArgParse.get("mode") == "eval")
    dataPath = g_ArgParse.get("dataPath") 
    robertaA = Roberta()
    robertaB = Roberta()
    robertaD = Roberta()
    robertaG = Roberta()
    robertaH = Roberta()

    dataset = Dataset(
        dataPath, 
        robertaA, robertaB, robertaD, robertaG, robertaH,
        eval_mode=eval_mode, devSetPath=g_ArgParse.get("devSetPath"), trainSetPath=g_ArgParse.get("trainSetPath")
    )
    dataset.printSetDistribution()

    modelA = VersionAAffectClassifier(dataset.number_of_users)
    modelB = VersionBAffectClassifier(dataset.number_of_users)
    modelD = VersionDAffectClassifier(dataset.number_of_users)
    modelG = VersionGAffectClassifier(dataset.number_of_users)
    modelH = VersionHAffectClassifier(dataset.number_of_users)

    if(eval_mode):
        print("Working in eval mode. Evaluating presaved models.")
        checkpoint = torch.load(g_ArgParse.get("modelSaveSpecific"), map_location="cpu")
        modelA.load_state_dict(checkpoint["modelA"])
        modelB.load_state_dict(checkpoint["modelB"])
        modelD.load_state_dict(checkpoint["modelD"])
        modelG.load_state_dict(checkpoint["modelG"])
        modelH.load_state_dict(checkpoint["modelH"])
        robertaA.getModel().load_state_dict(checkpoint["robertaA"])
        robertaB.getModel().load_state_dict(checkpoint["robertaB"])
        robertaD.getModel().load_state_dict(checkpoint["robertaD"])
        robertaG.getModel().load_state_dict(checkpoint["robertaG"])
        robertaH.getModel().load_state_dict(checkpoint["robertaH"])
        evaluate_arousal_mae(modelA, modelB, modelD, modelG, modelH, dataset)
    else:
        print("Working in training mode.")
        learning_rate = float(g_ArgParse.get("learningRate"))

        optimizerA = torch.optim.AdamW([
            {"params": modelA.parameters(), "lr": learning_rate},
            {"params": robertaA.getParameters(), "lr": 2e-5},
        ])
        optimizerB = torch.optim.AdamW([
            {"params": modelB.parameters(), "lr": learning_rate},
            {"params": robertaB.getParameters(), "lr": 2e-5},
        ])
        optimizerD = torch.optim.AdamW([
            {"params": modelD.parameters(), "lr": learning_rate},
            {"params": robertaD.getParameters(), "lr": 2e-5},
        ])

        optimizerG = torch.optim.AdamW([
            {"params": modelG.parameters(), "lr": learning_rate},
            {"params": robertaG.getParameters(), "lr": 2e-5}
        ])

        optimizerH = torch.optim.AdamW([
            {"params": modelH.parameters(), "lr": learning_rate},
            {"params": robertaH.getParameters(), "lr": 2e-5},  # separate encoder for H
        ])
    
        dataset.writeOutDevSet()
        dataset.writeOutTrainSet()

        trainingLoop(
            modelA, modelB, modelD, modelG, modelH,
            optimizerA, optimizerB, optimizerD, optimizerG, optimizerH, 
            dataset
        )




if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
