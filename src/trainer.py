import logging
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.callbacks.base_callback import BaseCallback
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.save_checkpoints import SaveCheckpoints
from src.metric_accumulator import MetricAccumulator
from src.metrics import BaseMetric


class Trainer(object):
    def __init__(
        self,
        *,
        model,
        optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        main_metric: BaseMetric,
        validation_dataloader: torch.utils.data.DataLoader = None,
        device="cpu",
        loss: Optional = None,
        scheduler: Optional = None,
        max_grad_norm: Optional[float] = None,
        n_epochs: int = 4,
        callbacks: Optional[List[BaseCallback]] = None,
        metrics: Optional[List[BaseMetric]] = None,
        tensorboard_writer: Optional[SummaryWriter] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_dataloader
        self.val_loader = validation_dataloader
        self.device = device
        self.model = self.model.to(self.device)
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.main_metric = main_metric
        self.metrics = metrics
        self.tensorboard_writer = tensorboard_writer
        self.metric_accumulator = MetricAccumulator()
        self.loss = loss
        if self.loss is not None:
            self.loss.to(self.device)

    def fit(self):
        logging.info("Training started...")
        # Store the average loss after each epoch so we can plot them.
        train_losses, validation_losses = [], []

        for n in range(1, self.n_epochs + 1):
            logging.info(f"Fitting {n} epoch.")
            train_loss = self.training_step(n)
            val_loss = self.validation_step(n)

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar("Loss/train", train_loss, n)
                self.tensorboard_writer.add_scalar("Loss/val", val_loss, n)
            train_losses.append(train_loss)
            validation_losses.append(val_loss)

    def training_step(self, epoch_number):
        train_loss = 0
        n_samples = len(self.train_loader)

        # Put the model into training mode.
        self.model.train()

        # Training loop
        for step, batch in tqdm(enumerate(self.train_loader)):
            batch = {t: batch[t].to(self.device) for t in batch}
            self.model.zero_grad()

            outputs = self.model(
                batch["input_ids"],
                token_type_ids=batch.get("token_type_ids"),
                attention_mask=batch["attention_masks"],
                labels=batch["labels"],
            )
            # get the loss
            if self.loss:
                # if loss is custom
                loss = self.loss(outputs[1], batch["labels"])
            else:
                # if self.loss is None and we use bert loss
                loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            self.optimizer.zero_grad()
            loss.backward()
            # track train loss
            train_loss += loss.item()

            # This is to help prevent the "exploding gradients" problem.
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.max_grad_norm,
                )
            # update parameters
            self.optimizer.step()

            # Update the learning rate.
            self.scheduler.step()

        logging.info("Train loss: %.2f" % (train_loss / n_samples))

        return train_loss / n_samples

    def validation_step(self, epoch_number):
        val_loss = 0
        n_samples = len(self.val_loader)
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                batch = {t: batch[t].to(self.device) for t in batch}

                # Forward pass, calculate logit predictions.
                # This will return the logits
                # rather than the loss because we have not provided labels.
                outputs = self.model(
                    batch["input_ids"],
                    token_type_ids=batch.get("token_type_ids"),
                    attention_mask=batch["attention_masks"],
                    labels=batch["labels"],
                )

                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy()
                label_ids = batch["labels"].to("cpu").numpy().tolist()
                if self.loss:
                    # if use custom loss
                    val_loss += self.loss(outputs[1], batch["labels"]).mean().item()
                else:
                    val_loss += outputs[0].mean().item()
                prediction_labels = np.argmax(logits, axis=-1).tolist()

                self.main_metric(
                    ground_labels=label_ids,
                    pred_labels=prediction_labels,
                    input_mask=batch["attention_masks"].detach().cpu().numpy(),
                )

                for metric in self.metrics:
                    metric(
                        ground_labels=label_ids,
                        pred_labels=prediction_labels,
                        input_mask=batch["attention_masks"].detach().cpu().numpy(),
                    ),

            logging.info("Validation loss: %.2f" % (val_loss / n_samples))

        # calculating main training metric and log it
        main_metric = self.main_metric.calculate()
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(
                f"Val/MainMetric/{self.main_metric.name}", main_metric, epoch_number
            )
        logging.info("Main metric %s is %.4f" % (self.main_metric.name, main_metric))

        # logging other metrics
        for metric in self.metrics:
            metric_value = metric.calculate()
            if isinstance(metric_value, float):
                logging.info("Metric %s is %.4f" % (metric.name, metric_value))
            else:
                logging.info("Metric %s is \n %s" % (metric.name, metric_value))
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(
                    f"Val/Metric/{metric.name}", metric_value, epoch_number
                )

        for callback in self.callbacks:
            if isinstance(callback, SaveCheckpoints) and callback.only_best:
                callback(n_epoch=epoch_number, metric=main_metric)
            if isinstance(callback, EarlyStopping):
                callback(val_loss / n_samples)
        return val_loss / n_samples
