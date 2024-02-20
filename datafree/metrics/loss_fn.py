import torch

def build_loss_fn(base_probs, tau=1.0, reduction='mean'):
    """Builds the loss function.
    Args:
        base_probs: Base probabilities to use in the logit-adjusted loss.
        tau: Temperature scaling parameter for the base probabilities.
        loss_type: the loss type for training. options:['lc', 'ce', 'bce']
    Returns:
        A loss function with signature loss(labels, logits).
    """
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    def bce_loss_fn(logits, labels, tau=tau):
        """ balanced cross entropy loss
        """
        logits = logits + tau * torch.log(base_probs + 1e-12) # avoid underflow
        loss = criterion(logits, labels)
        return loss
    return bce_loss_fn