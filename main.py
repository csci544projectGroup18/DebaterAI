import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy

from transformers import GPT2Model, GPT2Tokenizer
from transformers import MAMConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import EarlyStoppingCallback, IntervalStrategy

from torch.utils.data.dataset import Dataset


from transformers import Trainer, TrainingArguments, EvalPrediction, TrainerCallback
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import json
from datetime import datetime

from src.datasets.DebaterDataset import DebaterDataset
from src.utils import CustomDataCollator, custom_compute_metrics

class SequenceEncoderBlock(nn.Module):
    '''Sequence encoder block

    params:
        max_sequence_length: Maximum sequence length
        adapter_name: Adapter used for fine-tuning pre-trained encoder
        adapter_config: Adapter config
        cnn_output_channels: Number of output channels of the CNN(=dimension of the sequence embedding)
        cnn_window_size: Window size of the CNN
    '''
    def __init__(
            self,
            gpt2_hidden_size,
            max_sequence_length,
            cnn_output_channels,
            cnn_window_size,
            sequence_embedding_size
        ):
        super(SequenceEncoderBlock, self).__init__()

        #   CNN layer
        self.cnn = nn.Conv1d(
            in_channels=gpt2_hidden_size * 2,
            out_channels=cnn_output_channels,
            kernel_size=cnn_window_size,
            padding=int(cnn_window_size / 2)
        )

        #   Max pooling layer
        self.max_pooling = nn.MaxPool1d(kernel_size=max_sequence_length)

        #   Batch normalization layers
        self.word_embedding_bn = nn.BatchNorm1d(num_features=gpt2_hidden_size)
        self.encoder_bn = nn.BatchNorm1d(num_features=gpt2_hidden_size)
        self.pooling_bn = nn.BatchNorm1d(cnn_output_channels)
        self.linear = nn.Linear(sequence_embedding_size, sequence_embedding_size)

    def forward(self, outputs, attention_mask):
        '''Forward propagation

        params:
            input_ids: Tensor of shape (B, L) containing the input token IDs
            attention_mask: Tensor of shape (B, L) containing the attention mask
        '''
        #   Dimension notations:
        #   B: batch size
        #   L: sequence length
        #   H: hidden size
        #   C: number of output channels of the CNN (also the dimension of the sequence embedding)

        #   Get word embeddings and last hidden states from GPT-2

        word_embeddings = outputs.hidden_states[0]
        #   Dimension: (B, L, H)
        encoder_hidden_states = outputs.last_hidden_state
        #   Dimension: (B, L, H)

        #   Batch normalization
        bn_word_embeddings = self.word_embedding_bn(
            word_embeddings.permute(0, 2, 1)
        ).permute(0, 2, 1)
        bn_encoder_hidden_states = self.encoder_bn(
            encoder_hidden_states.permute(0, 2, 1)
        ).permute(0, 2, 1)

        #   Concatenate word embeddings and encoder hidden states
        concat_embeddings = torch.cat((bn_word_embeddings, bn_encoder_hidden_states), dim=-1)
        #   Dimension: (B, L, H * 2)

        #   Apply attention mask to the concatenated sequence representation
        #   The attetion mask is expanded to dimension (B, L, H * 2),
        #   matching the dimension of the concatenated sequence representation.
        #   The concatenated sequence representation is multiplied element-wise with the attention mask
        #   to zero out the padded positions.
        masked_concat_embeddings = concat_embeddings * \
            attention_mask.unsqueeze(-1).expand(concat_embeddings.shape)

        #   Apply CNN layer
        cnn_out = self.cnn(masked_concat_embeddings.permute(0, 2, 1))
        #   Dimension: (B, C, L)

        #   Apply max pooling layer
        pooled_output = self.max_pooling(cnn_out)
        #   Dimension: (B, C, 1)

        #   Apply batch normalization
        #   This is the final sequence embedding
        sequence_embedding = self.pooling_bn(pooled_output.squeeze(-1))
        #   Dimension: (B, C)

        return sequence_embedding

class StanceClassifier(nn.Module):
    '''Stance classifier

    params:
        parent_encoder: Sequence encoder block for parent comments
        child_encoder: Sequence encoder block for child comments
        context_encoder: Sequence encoder block for comment context
        sequence_embedding_size: Dimension of the sequence embedding
        ff_hidden_size: Hidden size of the feed-forward layer
        num_classes: Number of classes
    '''
    def __init__(
            self,
            # parent_encoder: SequenceEncoderBlock,
            # child_encoder: SequenceEncoderBlock,
            # context_encoder: SequenceEncoderBlock,
            adapter_name,
            adapter_config,
            loss_fn,
            sequence_embedding_size,
            ff_hidden_size,
            num_classes
        ):
        super(StanceClassifier, self).__init__()

        #   Pre-trained GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained("gpt2")

        #   Freeze GPT-2 pre-trained parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        #   Add adapter to GPT-2
        self.gpt2.add_adapter(adapter_name, config=adapter_config)
        self.gpt2.set_active_adapters(adapter_name)


        self.parent_encoder = SequenceEncoderBlock(
            gpt2_hidden_size=self.gpt2.config.hidden_size,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            cnn_output_channels=SEQUENCE_EMEBDDING_SIZE,
            cnn_window_size=CNN_WINDOW_SIZE,
            sequence_embedding_size = sequence_embedding_size
        )
        self.child_encoder = SequenceEncoderBlock(
            gpt2_hidden_size=self.gpt2.config.hidden_size,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            cnn_output_channels=SEQUENCE_EMEBDDING_SIZE,
            cnn_window_size=CNN_WINDOW_SIZE,
            sequence_embedding_size = sequence_embedding_size
        )
        self.context_encoder = SequenceEncoderBlock(
            gpt2_hidden_size=self.gpt2.config.hidden_size,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            cnn_output_channels=SEQUENCE_EMEBDDING_SIZE,
            cnn_window_size=CNN_WINDOW_SIZE,
            sequence_embedding_size = sequence_embedding_size
        )
        # self.sequence_encoder = sequence_encoder
        self.loss_fn = loss_fn

        #   Feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(sequence_embedding_size * 2, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, sequence_embedding_size),
            nn.ReLU(),
            nn.Linear(sequence_embedding_size, num_classes)
        )

    def forward(self, input_ids, attention_masks, labels=None):
        '''Forward propagation

        params:
            input_ids: list tensors of shape (B, L) containing the input token IDs
            attention_masks: list tensors of shape (B, L) containing the attention masks
            labels: Tensor of shape (B,) containing the labels
        '''
        #   Dimension notations:
        #   B: batch size
        #   S: dimension of the sequence embedding
        #   C: number of classes

        parent_outputs = self.gpt2(
            input_ids=input_ids[0],
            attention_mask=attention_masks[0],
            output_hidden_states=True
        )

        parent_embeddings = self.parent_encoder(parent_outputs, attention_mask = attention_masks[0])

        child_outputs = self.gpt2(
            input_ids=input_ids[1],
            attention_mask=attention_masks[1],
            output_hidden_states=True
        )

        child_embeddings = self.child_encoder(child_outputs, attention_mask = attention_masks[1])

        context_outputs = self.gpt2(
            input_ids=input_ids[2],
            attention_mask=attention_masks[2],
            output_hidden_states=True
        )

        context_embeddings = self.context_encoder(context_outputs, attention_mask = attention_masks[2])

        #   Dimension: 3 * (B, S)

        #   Create the combined sStanceClassifierloequence embedding for classification
        combined_embeddings = torch.cat(
            (parent_embeddings + context_embeddings, child_embeddings + context_embeddings),
            dim=-1
        )
        #   Dimension: (B, S * 2)

        #   Feed-forward layer
        logits = self.ff(combined_embeddings)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("DATASET_FILE", required=True, default = 'data/labeled_data.csv',
                        help="The path of csv data", 
                        type=str)
    parser.add_argument("--eval", action='store_true',
                        help="run evaluation")
    parser.add_argument("--ckpt",
                    help="checkpoint path, needed for eval",
                    type=str)
    args = parser.parse_args()

    # Config Env
    PROJECT_ROOT_DIR = os.getcwd()
    PRETRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, "models", "pretrained")
    DATASET_FILE = '/lab/xingrui/DebaterAI/data/labeled_data.csv'
    DATASET_FILE = args.DATASET_FILE

    #   Path to the directory where the pre-trained model will be saved.
    os.environ["HUGGINGFACE_HUB_CACHE"] = PRETRAINED_MODEL_DIR
    os.environ["TRANSFORMERS_CACHE"] = PRETRAINED_MODEL_DIR

    RESULTS_DIR = os.path.join(PROJECT_ROOT_DIR, "results")
    LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "logs")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   Hyperparameters for SequenceEncoder block.
    ADAPTER_NAME = "mam_adpater"
    ADAPTER_CONFIG = MAMConfig()

    # Training configuration
    MAX_SEQUENCE_LENGTH = 128
    SEQUENCE_EMEBDDING_SIZE = 1024
    CNN_WINDOW_SIZE = 9

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    FF_HIDDEN_SIZE = 4 * SEQUENCE_EMEBDDING_SIZE
    NUM_CLASSES = 3

    TRAINING_EPOCHS = 100
    BACTH_SIZE = 32
    LEARNING_RATE = 1e-5

    CLSModel = StanceClassifier(
        adapter_name=ADAPTER_NAME,
        adapter_config=ADAPTER_CONFIG,
        loss_fn=nn.CrossEntropyLoss(),
        sequence_embedding_size=SEQUENCE_EMEBDDING_SIZE,
        ff_hidden_size=FF_HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    )
    #   Optimizer and LR scheduler may need to be changed based on actual performance
    #   This is the default setting from the Trainer implementation
    optimizer = AdamW(CLSModel.parameters(), lr=LEARNING_RATE)
    lr_scheduler = LambdaLR(optimizer, lambda epoch: 1 / (epoch / 5000 + 1))

    # add dataset
    train_dataset = DebaterDataset(DATASET_FILE, is_test=False)
    eval_dataset = DebaterDataset(DATASET_FILE, is_test=True)

    MyCollator = CustomDataCollator(tokenizer, MAX_SEQUENCE_LENGTH)

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        logging_dir=LOG_DIR,
        logging_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=5,
        metric_for_best_model = 'f1',
        load_best_model_at_end=True,
        num_train_epochs=TRAINING_EPOCHS,
        per_device_train_batch_size=BACTH_SIZE,
        per_device_eval_batch_size=BACTH_SIZE,
        remove_unused_columns=False
    )

    ckpt = torch.load('/lab/xingrui/DebaterAI/results/model_best/pytorch_model.bin')
    CLSModel.load_state_dict(ckpt)
    trainer = Trainer(
        model=CLSModel,
        args=training_args,
        train_dataset=train_dataset,     #   Change this to the training dataset
        eval_dataset=eval_dataset,      #   Change this to the evaluation dataset
        data_collator=MyCollator,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=custom_compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )

    if args.eval:
        evaluation = trainer.evaluate()
        print(evaluation)
    else:
        train_result = trainer.train()
        trainer.save_model(f'{RESULTS_DIR}/model_best/')

        log_history = trainer.state.log_history
        time_format = time_format = "%Y-%m-%dT%H_%M_%S"
        with open(os.path.join(LOG_DIR, f"train_history_{datetime.utcnow().strftime(time_format)}.txt"), 'w') as train_h_output:
            train_h_output.write(json.dumps(log_history, indent=4))
        with open(os.path.join(LOG_DIR, f"train_result_{datetime.utcnow().strftime(time_format)}.txt"), 'w') as train_output:
            train_output.write(json.dumps(train_result, indent=4))

