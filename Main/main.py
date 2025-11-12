#!../bin/python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ClassDefinition.Utils import Logger, ArgumentParser
from losses import build_criterion, compute_single_task_loss
required_arguments=["testRequiredArgument1"]
optional_arguments=[]
g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print=g_Logger.print

USAGE="""
main.py testRequiredArgument1=<dummyValue>
"""

def initialize(inputArguments):
    g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)

def main(inputArguments):
    """
    run Valence OR Arousal in separate runs, controlled by an input tag:
      - task=valence  -> 5-class classification over [-2, 2] bins
      - task=arousal  -> 3-class classification over [0, 2] bins
    """

    # 1. Parse args and set variables (task, model_name, batch_size, epochs, lr, max_length, etc.)
    #   ex. python main.py task=valence max_length=256 batch_size=16 epochs=4 (optional: lr_enc=2e-5 lr_head=1e-3)

    initialize(inputArguments)
    print(g_ArgParse.get("testRequiredArgument1"))


    # 2. Load training CSV data (train_subtask1.csv)
    #   - required columns: user_id, text, valence (float in [-2,2]), arousal (float in [0,2])
    #   - drop NaNs, basic whitespace cleanup
    from data_ingest import ingest
    entries = ingest()

    # 3. Define bins for classification labels:
    #    - Valence: 5 bins over [-2, 2] -> class ids {0..4}
    #    - Arousal: 3 bins over [ 0, 2] -> class ids {0..2}
    for e in entries:
        #Compute class ids for valence and arousal and update the Entry object.
        e.valence_class = entry.valence + 2 #Range 0,1,2,3,4
        e.arousal_class = entry.arousal #Range 0,1,2

    # 4. Preprocess -> tokenize
    #   - use Hugging Face tokenzier for RoBERTa (padding=True, truncation=True, max_length=128-256)
    #   - create dataset/dataloader and return input_ids, attention_mask, y_valence_class, y_arousal_class


    # 5. Build model
    #   - Encoder: RoBERTa
    #   - build one head for the chosen task


    # 6. Loss and optimizer


    # 7. Training loop
    #   - for each batch:
    #       - forward -> logits [B, K]
    #       - loss, logs = compute_single_task_loss(logits, y_class, criterion)
    #       - backward, clip, optimizer.step(), scheduler.step()
    #       - log loss


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
