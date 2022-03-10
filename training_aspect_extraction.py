import argparse
import sys
import logging
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from src.callbacks import EarlyStopping
from src.callbacks import SaveCheckpoints
from src.datasets import AspectExtractionDataset
from src.metrics import SequenceAccuracyScore
from src.metrics import SequenceF1Score
from src.metrics import SequencePrecisionScore
from src.metrics import SequenceRecallScore
from src.trainer import Trainer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT model for aspect extraction task.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=10e-3)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--full_finetuning", type=bool, default=True)
    parser.add_argument("--path_to_bert", type=str, default="./models/bert_lm_pretrained")
    parser.add_argument("--path_to_checkpoints", type=str, default="./models/checkpoints/")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.path_to_bert, do_lower_case=False)
    logger.info("Tokenizer is loaded")

    # solving aspect-category sentence classification
    train_dataset = AspectExtractionDataset(
        "./data/raw/SentiRuEval_rest_markup_train.xml", tokenizer=tokenizer
    )
    test_dataset = AspectExtractionDataset(
        "./data/raw/SentiRuEval_rest_markup_test.xml",
        tokenizer=tokenizer,
        label_to_id=train_dataset.label_to_id,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    valid_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    logger.info("Initializing BertForTokenClassification from pretrained BERT LM")
    model = BertForTokenClassification.from_pretrained(
        args.path_to_bert,
        num_labels=len(train_dataset.label_to_id),
        output_attentions=False,
        output_hidden_states=False,
    )

    if args.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8,
    )

    # add a scheduler to linearly reduce the learning rate throughout the epochs
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info("CUDA is available")
        model.cuda()

    tensorboard_writer = SummaryWriter(log_dir="logs/fit")
    trainer = Trainer(
        model=model,
        main_metric=SequenceF1Score(len(train_dataset.label_to_id)),
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
        device=device,
        scheduler=scheduler,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        callbacks=[
            SaveCheckpoints(model, args.path_to_checkpoints),
            EarlyStopping(),
        ],
        metrics=[
            SequenceAccuracyScore(),
            SequenceRecallScore(len(train_dataset.label_to_id)),
            SequencePrecisionScore(len(train_dataset.label_to_id)),
        ],
        tensorboard_writer=tensorboard_writer,
    )
    trainer.fit()
