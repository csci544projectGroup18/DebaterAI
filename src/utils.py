
import os
import numpy as np
import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2Tokenizer
from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


#   Collate function for the combined classifier
class CustomDataCollator:
    def __init__(self, tokenizer: GPT2Tokenizer, MAX_SEQUENCE_LENGTH):
        self.tokenizer = tokenizer
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    def __call__(self, batch):
        #   Dimension notations:
        #   B: batch size
        #   L: sequence length

        parent_comment = [item['parent_comment'] for item in batch]
        child_comment = [item['child_comment'] for item in batch]
        context = [item['context'] for item in batch]
        labels = [item['label'] for item in batch]

        #   Tokenize the input sequences
        parent_tokenized = self.tokenizer(
            parent_comment,
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        child_tokenized = self.tokenizer(
            child_comment,
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        context_tokenized = self.tokenizer(
            context,
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

        labels = torch.tensor(labels)

        input_ids = [
            parent_tokenized['input_ids'],
            child_tokenized['input_ids'],
            context_tokenized['input_ids']
        ]
        attention_masks = [
            parent_tokenized['attention_mask'],
            child_tokenized['attention_mask'],
            context_tokenized['attention_mask']
        ]

        return {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}


def custom_compute_metrics(eval_pred: EvalPrediction) -> dict:
    '''Compute metrics for the combined classifier

    params:
        eval_pred: EvalPrediction object
    '''
    #   Dimension notations:
    #   B: batch size
    #   C: number of classes

    #   Convert logits to predictions
    preds = np.argmax(eval_pred.predictions, axis=1)

    min_len = min(len(eval_pred.label_ids), len(preds))
    #   Dimension: (B,)

    #   Compute precision, recall, and F1 score
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     eval_pred.label_ids[:min_len], preds[:min_len], average="weighted"
    # )
    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_pred.label_ids, preds, average="weighted"
    )

    #   Compute confusion matrix
    # cm = confusion_matrix(eval_pred.label_ids[:min_len], preds[:min_len])
    cm = confusion_matrix(eval_pred.label_ids, preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sub_accuracy": (cm.diagonal() / cm.sum(axis=1)).tolist(),
        "accuracy": cm.diagonal().sum() / cm.sum()
    }

#   For debug use only, not used in the actual training.
def custom_logits_preprocessing(logits: torch.Tensor, labels: torch.Tensor):
    '''Preprocess logits before computing loss

    params:
        logits: logits from the model
        labels: labels for the batch
    '''
    print("\nInside custom_logits_preprocessing")
    print("Shape of logits: {}".format(logits.shape))
    print("Actual logits: {}".format(logits))
    print("Shape of labels: {}".format(labels.shape))
    print("Actual labels: {}".format(labels))
    print("Out of custom_logits_preprocessing\n")
    return logits
