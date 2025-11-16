#!../bin/python3
import sys, os, random
import torch
import torch.nn.functional as F

SEMROOT = os.environ['SEMROOT']
sys.path.append(SEMROOT)

from ClassDefinition.Utils import Logger, ArgumentParser
from ClassDefinition.Entry import Entry
from ClassDefinition.Roberta import Roberta
from ClassDefinition.Dataset import Dataset, Batch
from ClassDefinition.ArousalClassifier import ArousalClassifier
from losses import build_criterion, compute_single_task_loss

required_arguments = []
optional_arguments = {
    "dataPath": f"{SEMROOT}/Data/TRAIN_RELEASE_3SEP2025/train_subtask1.csv",
    "numEpochs": 5,  # 60?
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
    dataset.shuffle()
    batch_size = int(g_ArgParse.get("batchSize"))
    num_epochs = int(g_ArgParse.get("numEpochs"))

    dataset.setTrainBatchList(batch_size)
    train_batch_list = dataset.getTrainBatchList()

    criterion = build_criterion()
    number_of_batches = len(train_batch_list)
    print("Beginning training loop")

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for j in range(number_of_batches):
            if j % 10 == 0:
                print(
                    f"\tEpoch:{epoch+1}/{num_epochs}\tbatch:{j}/{number_of_batches}"
                )

            batch = train_batch_list[j]

            # all CLS embeddings of each entry in batch
            features = batch.getFeatures()  # [B, 768]

            # the labels for arousals for the batch
            arousalLabels = batch.arousalLabelList  # [B] long

            optimizer.zero_grad()

            # get logits of the model (should output 3-way)
            logits = model(features)  # [B, 3]

            # get loss
            loss, log = compute_single_task_loss(logits, arousalLabels, criterion)

            # backprop loss
            loss.backward()

            # TODO could implement gradient clipping here (small, but helps keep training stable)

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
          - features -> logits
          - probs = softmax(logits)
          - y_hat_float = sum(probs * bin_centers)
      - Compute MAE vs the original float arousal labels
    """
    model.eval()
    batch_size = int(g_ArgParse.get("batchSize"))

    # Here we treat the 3 classes as centered at arousal values {0.0, 1.0, 2.0}
    # which matches how you discretized arousal_class.
    bin_centers = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    total_abs_error = 0.0
    total_count = 0

    with torch.no_grad():
        dev_entries = dataset.devSet

        for i in range(0, len(dev_entries), batch_size):
            batch = Batch(dev_entries[i : i + batch_size], dataset.roberta)

            # Features for this dev batch (reuses existing Batch logic & Roberta)
            features = batch.getFeatures()  # [B, 768]

            # Logits and probabilities
            logits = model(features)          # [B, 3]
            probs = F.softmax(logits, dim=1)  # [B, 3]

            # Expected arousal value per example using bin centers
            # y_hat_float: [B]
            y_hat_float = (probs * bin_centers).sum(dim=1)

            # True continuous arousal labels from the entries
            true_arousal = torch.tensor(
                [float(e.arousal) for e in batch.entryList],
                dtype=torch.float32,
            )

            abs_errors = (y_hat_float - true_arousal).abs()
            total_abs_error += abs_errors.sum().item()
            total_count += abs_errors.numel()

    mae = total_abs_error / max(total_count, 1)
    print(f"Dev MAE (arousal): {mae:.4f}")
    return mae


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
    from data_ingest import ingest

    entries = ingest(g_ArgParse.get("dataPath"))
    print(f"{len(entries)} entries ingested")

    # 3. Define bins for classification labels:
    #    - Valence: 5 bins over [-2, 2] -> class ids {0..4}
    #    - Arousal: 3 bins over [ 0, 2] -> class ids {0..2}
    for e in entries:
        # Compute class ids for valence and arousal and update the Entry object.
        e.valence_class = int(round(float(e.valence))) + 2
        e.valence_class = max(0, min(4, e.valence_class))  # 0..4
        e.arousal_class = int(round(float(e.arousal)))
        e.arousal_class = max(0, min(2, e.arousal_class))  # 0..2

    # 4. Preprocess -> tokenize
    #   - use Hugging Face tokenizer for RoBERTa
    #   - create dataset/dataloader and return input_ids, attention_mask, y_valence_class, y_arousal_class
    # dataset = Dataset(entries)  # splits into training and dev set
    roberta = Roberta()            # create only once
    dataset = Dataset(entries, roberta)

    # 5. Build model
    model = ArousalClassifier()

    # 6. Loss and optimizer
    learning_rate = float(g_ArgParse.get("learningRate"))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
