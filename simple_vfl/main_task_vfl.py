"""
MainTaskVFL Class

This class implements the main Vertical Federated Learning (VFL) task, which handles the training and evaluation of
a machine learning model across multiple parties. The class provides the framework to load datasets, run
distributed training and evaluate the model's performance.

Key functionalities:
- Initializes key parameters from the configuration, including the number of classes, device settings and number of
  epochs.
- Loads datasets and creates data loaders for training, validation, and testing.
- Instantiates a federated model based on the specified configuration.
- Supports early stopping based on validation performance to avoid overfitting.
- Implements a main training loop with batch-wise processing, local and global updates, and per-class metric tracking.
- Provides an evaluation method to compute metrics like loss, accuracy, AUC, and per-class performance on
  validation and test sets.
- Supports logging to TensorBoard and handling model checkpointing.
"""

import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAUROC

from .dataset.load_dataset import load_datasets
from .models import complete_model as model_zoo
from .utils.basic_functions import seed_worker
from .utils.early_stopping import EarlyStopping

# Use the same logger as in the main file
logger = logging.getLogger(__name__)


class MainTaskVFL:
    def __init__(self, args: Any):
        """
        Initialize the main VFL task with the provided arguments.
        - Setup key parameters like the number of classes, device and number of epochs.
        - Load datasets and create dataloaders for training, validation, and testing.
        - Instantiate the complete model based on the configuration.
        - Optionally, set up early stopping if specified in the arguments.
        """
        self.args = args
        self.k = args.k
        self.device = args.runtime.device
        self.dataset_name = args.dataset.dataset_name
        self.epochs = args.epochs
        self.num_classes = args.dataset.num_classes
        self.parties_data = None
        self.label = None
        self.criterion = self._initialize_criterion(args)

        # Load datasets and create dataloaders
        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(args)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.runtime.num_workers,
            shuffle=True,
            worker_init_fn=seed_worker,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            num_workers=args.runtime.num_workers,
            shuffle=False,
            worker_init_fn=seed_worker,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            num_workers=args.runtime.num_workers,
            shuffle=False,
            worker_init_fn=seed_worker,
        )

        # Initialize the model based on the config
        self.complete_model = getattr(model_zoo, args.model)(args).to(self.device)

        # Initialize early stopping if configured
        if args.get("early_stopping") is not None:
            self.early_stopping = EarlyStopping(self.args.early_stopping)

    def _initialize_criterion(self, args: Any) -> torch.nn.Module:
        """
        Initializes the loss function (CrossEntropyLoss) based on the provided arguments.
        If class weighting is specified, it sets class-weighted cross-entropy loss.
        Label smoothing is also applied if specified in the config.
        """
        label_smoothing = args.get("label_smoothing") or 0.0
        class_weight = args.get("class_weight")
        if class_weight:
            weights = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
            weights = weights / weights.sum()
            return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train_batch(self) -> tuple[float, int, dict[int, dict[str, float]]]:
        """
        Train the model for a single batch of data.
        Performs the forward pass, the global update, and computes per-class metrics for the batch.
        """
        # ======== Forward pass ===========

        _, pred = self.complete_model(self.parties_data)

        # Compute predictions and correct classifications
        _, predicted = pred.max(1)
        correct_train_batch = predicted.eq(self.label).sum().item()

        # Compute loss for the batch
        loss = self.criterion(pred, self.label)
        batch_loss = loss.item()

        # Backpropagation and optimization
        self.complete_model.zero_grad()
        loss.backward()
        self.complete_model.optimizer.step()

        # Calculate per-class metrics
        per_class_metrics = defaultdict(lambda: {"loss": 0, "correct": 0, "total": 0})
        for class_idx in range(self.num_classes):
            class_mask = self.label == class_idx
            if class_mask.sum() > 0:
                per_class_metrics[class_idx]["correct"] += predicted[class_mask].eq(self.label[class_mask]).sum().item()
                per_class_metrics[class_idx]["total"] += class_mask.sum().item()
                per_class_metrics[class_idx]["loss"] += (
                    self.criterion(pred[class_mask], self.label[class_mask]).item() * class_mask.sum().item()
                )

        return batch_loss, correct_train_batch, dict(per_class_metrics)

    def train(self):
        """
        Main training loop over all epochs.
        Tracks and logs training and validation performance (loss, accuracy, AUC, and per-class metrics).
        Performs early stopping and model saving/loading based on the configuration.
        """
        writer = SummaryWriter()  # for tensorboard
        self.num_total_comms = 0
        self.current_epoch = 0
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        running_per_class_metrics = defaultdict(lambda: {"loss": 0, "correct": 0, "total": 0})
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            i = -1

            for batch_data, batch_labels in self.train_loader:
                self.label = batch_labels.to(self.device)
                self.parties_data = [data.to(self.device) for data in batch_data]
                i += 1
                self.complete_model.train()

                batch_loss, correct_train_batch, batch_per_class_metrics = self.train_batch()
                running_loss += batch_loss * self.label.shape[0]
                correct_train += correct_train_batch
                total_train += self.label.shape[0]

                # Update running per-class metrics
                for class_idx, metrics in batch_per_class_metrics.items():
                    running_per_class_metrics[class_idx]["loss"] += metrics["loss"]
                    running_per_class_metrics[class_idx]["correct"] += metrics["correct"]
                    running_per_class_metrics[class_idx]["total"] += metrics["total"]

                self.num_total_comms = self.num_total_comms + 1

            # Perform validation after each epoch
            val_loss, val_accuracy, val_auc, val_per_class_metrics = self.evaluate("val")
            self._validate_and_log(
                i_epoch,
                running_loss / total_train,
                correct_train / total_train,
                running_per_class_metrics,
                writer,
                val_loss,
                val_accuracy,
                val_auc,
                val_per_class_metrics,
            )

            if self.__early_stop(val_loss):
                break

            # Step scheduler if configured
            if "scheduler" in self.args:
                self.complete_model.scheduler_step(val_loss)

            # Reset running metrics for the next epoch
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            running_per_class_metrics.clear()

        # Optionally save the final model
        if self.args.runtime.save_model:
            self.complete_model.save_model("trained.pt")

        # Restore the best model saved by early stopping before testing
        if hasattr(self, "early_stopping"):
            self.complete_model.load_model("best_model.pt")

        # Final evaluation on the test set
        test_loss, test_accuracy, test_auc, test_per_class_metrics = self.evaluate("test")
        logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
        writer.add_scalar("Loss/test", test_loss, self.epochs)
        writer.add_scalar("Accuracy/test", test_accuracy, self.epochs)
        writer.add_scalar("AUC/test", test_auc, self.epochs)
        self._log_per_class_metrics(test_per_class_metrics, {}, "test")

        writer.close()

    def __early_stop(self, val_loss: float) -> bool:
        """
        Checks if early stopping is triggered based on validation loss.
        Returns True if training should stop early.
        """
        if hasattr(self, "early_stopping"):
            self.early_stopping(val_loss, self.complete_model)
            if self.early_stopping.early_stop:
                logger.info("Early stopping")
                return True
        return False

    def _validate_and_log(
        self,
        i_epoch: int,
        train_loss: float,
        train_acc: float,
        running_per_class_metrics: dict[int, dict[str, float]],
        writer: SummaryWriter,
        val_loss: float,
        val_accuracy: float,
        val_auc: float,
        val_per_class_metrics: dict[int, dict[str, float]],
    ):
        """
        Log training and validation metrics to TensorBoard and the logger.
        """
        writer.add_scalar("Loss/train", train_loss, i_epoch)
        writer.add_scalar("Accuracy/train", train_acc, i_epoch)
        writer.add_scalar("Loss/val", val_loss, i_epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, i_epoch)
        writer.add_scalar("AUC/val", val_auc, i_epoch)

        self._log_per_class_metrics(val_per_class_metrics, running_per_class_metrics)

        logger.info(
            f"Epoch {i_epoch + 1}, {(i_epoch + 1) / self.epochs * 100:.2f}% \t "
            f"train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} "
            f"val_loss:{val_loss:.4f} val_acc:{val_accuracy:.4f} val_auc:{val_auc:.4f}"
        )

    def _log_per_class_metrics(
        self, val_metrics: dict[int, dict[str, float]], train_metrics: dict[int, dict[str, float]], split: str = "val"
    ):
        """
        Log per-class loss and accuracy.
        """
        for class_idx, metrics in val_metrics.items():
            if metrics["total"] > 0:
                logger.debug(
                    f"class_{class_idx} {split}_loss: {metrics['loss'] / metrics['total']:.4f}, "
                    f"{split}_accuracy: {metrics['correct'] / metrics['total']:.4f}"
                )
        for class_idx, metrics in train_metrics.items():
            if metrics["total"] > 0:
                logger.debug(
                    f"class_{class_idx} train_loss: {metrics['loss'] / metrics['total']:.4f}, "
                    f"train_accuracy: {metrics['correct'] / metrics['total']:.4f}"
                )

    def evaluate(self, mode: str = "test") -> tuple[float, float, float, dict[int, dict[str, float]]]:
        """
        Evaluates the model's performance on the validation or test set.
        Computes the loss, accuracy, AUC, and per-class metrics.
        """
        self.complete_model.eval()

        loader = self.val_loader if mode == "val" else self.test_loader
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_preds = []
        all_labels = []

        per_class_metrics = defaultdict(lambda: {"loss": 0, "correct": 0, "total": 0})

        with torch.no_grad():
            for batch_data, batch_labels in loader:
                self.label = batch_labels.to(self.device)
                self.parties_data = [data.to(self.device) for data in batch_data]

                _, pred = self.complete_model(self.parties_data)

                # Compute predictions and correct classifications
                _, predicted = pred.max(1)
                correct_preds += predicted.eq(self.label).sum().item()
                total_preds += self.label.size(0)

                # Compute loss
                loss = self.criterion(pred, self.label)
                running_loss += loss.item() * self.label.size(0)

                # Collect predictions and labels for AUC computation
                all_preds.append(pred.detach())
                all_labels.append(self.label)

                # Update per-class metrics
                for class_idx in range(self.num_classes):
                    class_mask = self.label == class_idx
                    if class_mask.sum() > 0:
                        per_class_metrics[class_idx]["correct"] += (
                            predicted[class_mask].eq(self.label[class_mask]).sum().item()
                        )
                        per_class_metrics[class_idx]["total"] += class_mask.sum().item()
                        per_class_metrics[class_idx]["loss"] += (
                            self.criterion(pred[class_mask], self.label[class_mask]).item() * class_mask.sum().item()
                        )

        # Concatenate predictions and labels across batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute AUC across all classes
        auroc = MulticlassAUROC(num_classes=self.num_classes).to(self.device)
        auc_score = auroc(all_preds, all_labels).mean().item()

        # Compute accuracy
        accuracy = correct_preds / total_preds
        return running_loss / total_preds, accuracy, auc_score, dict(per_class_metrics)
