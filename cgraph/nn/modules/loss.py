from cgraph import Node
from cgraph.autograd.functions import CrossEntropyBackward
import cgraph.nn.functional as F

class LossFunction:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __repr__(self):
        return f"LossFunction(name={self.name})"
    
class CrossEntropyLoss(LossFunction):
    def __init__(self):
        super().__init__("CrossEntropyLoss")

    def __call__(self, logits: Node, targets: Node) -> Node:
        """
        Computes the cross-entropy loss between logits and targets.
        Args:
            logits: Predicted logits (unnormalized scores).
            targets: True labels (class indices).
        Returns:
            Loss value.
        """

        # print("Logits:", logits)
        # print("Targets:", targets)

        if logits.ndim == 1:
            if targets.ndim == 1:
                num_classes = logits.shape[0]
                targets = F.one_hot_encode(targets, num_classes)
            else:
                assert logits.shape == targets.shape, "Logits and targets must have the same shape for cross-entropy loss."

            softmax = logits.softmax(dim=-1)
            log_probs = softmax.log()
            loss = -(log_probs * targets).sum()
        elif logits.ndim == 2:
            if targets.ndim == 1:
                num_classes = logits.shape[1]
                targets = F.one_hot_encode(targets, num_classes)
            else:
                assert logits.shape[0] == targets.shape[0], "Logits and targets must have the same batch size for cross-entropy loss."
                assert logits.shape[1] == targets.shape[1], "Logits and targets must have the same number of classes for cross-entropy loss."

            batch_size = logits.shape[0]
            softmax = logits.softmax(dim=-1)
            log_probs = softmax.log()
            loss = -(log_probs * targets).sum() / batch_size # Average over batch size

        # print(targets)

        if not loss.requires_grad:
            loss.requires_grad = True
            loss.grad_fn = CrossEntropyBackward(logits, targets)

        return loss
