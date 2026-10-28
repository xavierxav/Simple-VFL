import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when the validation loss stops improving."""

    def __init__(self, early_stopping_config):
        self.patience = early_stopping_config.patience
        self.delta = early_stopping_config.delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.debug("EarlyStopping counter: %s out of %s", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model) -> None:
        """Save the model when the validation loss decreases."""
        logger.debug(
            "Validation loss decreased (%.6f --> %.6f). Saving model ...",
            self.val_loss_min,
            val_loss,
        )
        model.save_model("best_model.pt")
        self.val_loss_min = val_loss
