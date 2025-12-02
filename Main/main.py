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
from ClassDefinition.AffectClassifier import VersionAAffectClassifier, VersionBAffectClassifier, VersionDAffectClassifier, VersionGAffectClassifier, VersionHAffectClassifier
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
    print("Beginning training loop")
        
    for epoch in range(num_epochs):
        dataset.shuffle()
        train_losses = []
        evaluate_arousal_mae(modelA, modelB, modelD, modelG, modelH, dataset)
        modelA.train()
        modelB.train()
        modelD.train()
        modelG.train()
        modelH.train()
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

            # CLS embeddings
            cls_embeddingsA = batch.getClsEmbeddingsA()
            cls_embeddingsB = batch.getClsEmbeddingsB()
            cls_embeddingsD = batch.getClsEmbeddingsD()
            cls_embeddingsG = batch.getClsEmbeddingsG()
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

            optimizerA.zero_grad()
            optimizerB.zero_grad()
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            optimizerH.zero_grad()

            # model predictions (REMOVED LEXICON)
            predictionsA = modelA(cls_embeddingsA, user_indices, is_words)
            predictionsB = modelB(cls_embeddingsB, user_indices, is_words)
            predictionsD = modelD(cls_embeddingsD, user_indices, is_words)
            predictionsG = modelG(cls_embeddingsG, user_indices, is_words)
            predictionsH = modelH(cls_embeddingsG, user_indices, is_words)

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
            # lossG.backward() # # Version G is frozen — compute loss only for logging (NO backward)


            # Version H: same ordinal targets as G
            valence_loss_H = criterionH(predictionsH["valence"], valenceLabelsG)
            arousal_loss_H = criterionH(predictionsH["arousal"], arousalLabelsG)
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
            torch.nn.utils.clip_grad_norm_(modelH.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dataset.robertaG.getParameters(), 1.0)

            # optimizers
            optimizerA.step()
            optimizerB.step()
            optimizerD.step()
            optimizerG.step()
            optimizerH.step()

            train_losses.extend([lossA.item(), lossB.item(), lossD.item(), lossG.item(), lossH.item()])

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        print(f"Epoch {epoch+1}/{num_epochs} average train loss: {avg_train_loss:.4f}")


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

    with torch.no_grad():
        for dev_batch in dataset.getDevBatchList():
            cls_embeddingsA = dev_batch.getClsEmbeddingsA()
            cls_embeddingsB = dev_batch.getClsEmbeddingsB()
            cls_embeddingsD = dev_batch.getClsEmbeddingsD()
            cls_embeddingsG = dev_batch.getClsEmbeddingsG()
            user_indices = dev_batch.getUserIndices()
            is_words = dev_batch.getIsWords()

            # class labels
            arousalLabels = (dev_batch.arousalLabelList + 1.0).to(torch.long)
            valenceLabels = (dev_batch.valenceLabelList + 2.0).to(torch.long)

            # predictions (NO LEXICON)
            predictionsA = modelA(cls_embeddingsA, user_indices, is_words)
            predictionsB = modelB(cls_embeddingsB, user_indices, is_words)
            predictionsD = modelD(cls_embeddingsD, user_indices, is_words)
            predictionsG = modelG(cls_embeddingsG, user_indices, is_words)
            predictionsH = modelH(cls_embeddingsG, user_indices, is_words)

            predictionsAArousal = torch.round(predictionsA["arousal"] * 2).clamp(0, 2)
            predictionsAValence = torch.round(predictionsA["valence"] * 4).clamp(0, 4)

            predictionsBArousal = torch.round(
                torch.clamp(predictionsB["arousal"] + 1, min=0, max=2)
            )
            predictionsBValence = torch.round(
                torch.clamp(predictionsB["valence"] + 2, min=0, max=4)
            )

            predictionsDArousal = predictionsD["arousal"].argmax(dim=-1)
            predictionsDValence = predictionsD["valence"].argmax(dim=-1)

            predictionsGArousal = (torch.sigmoid(predictionsG["arousal"]) > 0.5).sum(dim=1)
            predictionsGValence = (torch.sigmoid(predictionsG["valence"]) > 0.5).sum(dim=1)

            predictionsHArousal = (torch.sigmoid(predictionsH["arousal"]) > 0.5).sum(dim=1)
            predictionsHValence = (torch.sigmoid(predictionsH["valence"]) > 0.5).sum(dim=1)

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

            arousal_predicted = torch.mode(arousalVotes, dim=0).values
            valence_predicted = torch.mode(valenceVotes, dim=0).values

            arousal_predictions.append(arousal_predicted)
            valence_predictions.append(valence_predicted)

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

            valence_labels.append(valenceLabels)
            arousal_labels.append(arousalLabels)

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

    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousalA = f1(target=arousal_labels.long(), preds=arousal_predictions_A.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValenceA = f1(target=valence_labels.long(), preds=valence_predictions_A.long())
    print(f"Dev F1 A (arousal): {f1ScoreArousalA:.4f}")
    print(f"Dev F1 A (valence): {f1ScoreValenceA:.4f}")

    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousalB = f1(target=arousal_labels.long(), preds=arousal_predictions_B.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValenceB = f1(target=valence_labels.long(), preds=valence_predictions_B.long())
    print(f"Dev F1 B (arousal): {f1ScoreArousalB:.4f}")
    print(f"Dev F1 B (valence): {f1ScoreValenceB:.4f}")

    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousalD = f1(target=arousal_labels.long(), preds=arousal_predictions_D.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValenceD = f1(target=valence_labels.long(), preds=valence_predictions_D.long())
    print(f"Dev F1 D (arousal): {f1ScoreArousalD:.4f}")
    print(f"Dev F1 D (valence): {f1ScoreValenceD:.4f}")

    f1 = F1Score(task='multiclass',num_classes=3, average='macro')
    f1ScoreArousalG = f1(target=arousal_labels.long(), preds=arousal_predictions_G.long())
    f1 = F1Score(task='multiclass',num_classes=5, average='macro')
    f1ScoreValenceG = f1(target=valence_labels.long(), preds=valence_predictions_G.long())
    print(f"Dev F1 G (arousal): {f1ScoreArousalG:.4f}")
    print(f"Dev F1 G (valence): {f1ScoreValenceG:.4f}")

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
    
    return (valence_mae, arousal_mae, f1ScoreArousal, f1ScoreValence, AccuracyArousal, AccuracyValence, PrecisionArousal, PrecisionValence, RecallArousal, RecallValence, CombinedF1)

def save_model_and_bins(model: torch.nn.Module):
    artifacts_dir = os.path.join(SEMROOT, "Artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    save_path = os.path.join(artifacts_dir, "arousal_head.pt")

    bin_centers = [0.0, 1.0, 2.0]
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

    robertaA = Roberta()
    robertaB = Roberta()
    robertaD = Roberta()
    robertaG = Roberta()

    dataset = Dataset(
        g_ArgParse.get("dataPath"), 
        g_ArgParse.get("lexiconLookupPath"), 
        robertaA, robertaB, robertaD, robertaG
    )
    dataset.printSetDistribution()

    modelA = VersionAAffectClassifier(dataset.number_of_users)
    modelB = VersionBAffectClassifier(dataset.number_of_users)
    modelD = VersionDAffectClassifier(dataset.number_of_users)
    modelG = VersionGAffectClassifier(dataset.number_of_users)
    modelH = VersionHAffectClassifier(dataset.number_of_users)

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
    # Freeze G completely (G is reference-only now) # TODO Sierra check
    for p in modelG.parameters():
        p.requires_grad = False
    for p in robertaG.getParameters():
        p.requires_grad = False

    optimizerG = torch.optim.AdamW([
        {"params": modelG.parameters(), "lr": learning_rate},
    ])

    # H is the only model allowed to update RobertaG
    optimizerH = torch.optim.AdamW([
        {"params": modelH.parameters(), "lr": learning_rate},
        {"params": robertaG.getParameters(), "lr": 2e-5},
    ])

    
    trainingLoop(
        modelA, modelB, modelD, modelG, modelH,
        optimizerA, optimizerB, optimizerD, optimizerG, optimizerH, 
        dataset
    )

    evaluate_arousal_mae(
        modelA, modelB, modelD, modelG, modelH, 
        dataset
    )

    save_model_and_bins(modelA)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
