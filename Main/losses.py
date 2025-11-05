import torch
import torch.nn as nn

"""
Build a single CrossEntropy criterion (because performing categorical prediction) for the selected task
- class_weights = torch.tensor([...]) or None
- Inputs to CE: logits [B, K]
- Targets: class ids [B] (dtype long), values in {0..K-1}
"""
def build_criterion(class_weights=None, label_smoothing: float = 0.0):
    # CrossEntropyLoss internally applies softmax to turn logits into probibilities and takes negative log-probability of correct class
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )


"""
Compute cross entropy loss for the single active task
Returns: scalar loss (for backward) and a dict for logging
"""
def compute_single_task_loss(
    logits,      # [B, K] from the selected head
    y_classes,   # [B] LongTensor class ids in {0..K-1}
    criterion: nn.Module
):
    loss = criterion(logits, y_classes) # negative log probability the model assigned to the correct class (to be back propagated)
 
    return loss, {"loss": loss.detach().item()}
