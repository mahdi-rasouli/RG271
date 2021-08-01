import argparse
import ast
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AdamW, DistilBertForSequenceClassification, DistilBertTokenizerFast

from dataloader import CustomDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "precision": precision,
        "recall": recall,
        "f1": 2 * (precision * recall) / (precision + recall),
    }


def load_data(input_dir, batch_size, set_type, seed=None):
    df = pd.read_csv(Path(input_dir, f"{set_type}.csv"))
    df = df.loc[df["complaint"].isin([0, 1])]
    df = df[["content", "complaint"]]
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Perform a train/val split and setup data for torch DataLoader
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"].values.tolist(),
        df["complaint"].values.astype(int).tolist(),
        test_size=0.2,
        random_state=seed,
    )

    encodings = tokenizer(df["content"].values.tolist(), truncation=True, padding=True)
    dataset = CustomDataset(encodings, df["complaint"].values.astype(int).tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def save_model(model, model_dir, model_name="model.pth"):
    model.eval()
    return torch.save(model.cpu().state_dict(), Path(model_dir, model_name))


def validate(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true = []
    y_pred = []
    with torch.no_grad():
        total_loss = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()

            y_true += labels.to("cpu").numpy().flatten().tolist()
            y_pred += np.argmax(outputs[1].detach().cpu().numpy(), axis=1).flatten().tolist()

        logger.debug(f"Val Loss:{total_loss / len(val_loader):.4f};")

        metrics = calculate_metrics(y_true, y_pred)
        for metric, value in metrics.items():
            logger.debug(f"{metric.title()}:{value:2f};")


def train(args):
    """Configure and train a DistillBERT model.

    Args:
        dataset: DataFrame with columns 'Review' and 'Sentiment' to tokenize and use for training.
        args: CLI arguments with SageMaker and hyperparameter configuration.
    """

    # Setup classification model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)

    # Create DataLoaders for input data
    train_loader = load_data(args.data_dir, args.batch_size, "train", seed=args.seed)
    val_loader = load_data(args.data_dir, args.batch_size, "val", seed=args.seed)

    # Begin training
    for epoch in range(args.epochs):
        logger.debug(f"--- Starting epoch {epoch} ---")
        total_loss = 0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            optim.step()

        logger.debug(f"Train Loss:{total_loss / len(train_loader):.4f}")

        if val_loader is not None:
            validate(model, val_loader)

        save_model(model, args.model_dir, model_name=f"model_{epoch:04d}.pkl")

    return save_model(model, args.model_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "-i", "--input-data", required=True, type=str, help="URI to the labels in S3."
    # )
    parser.add_argument(
        "--epochs", required=True, type=int, help="Number of total epochs to run (default: 2)"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed to use.")

    # The parameters below retrieve their default values from SageMaker environment variables,
    # which are instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
