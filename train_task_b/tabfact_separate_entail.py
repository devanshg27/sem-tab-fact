import torch
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering, TapasConfig
import os
from torch import cuda
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score, confusion_matrix
import glob
import collections
import copy
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune TAPAS')
    parser.add_argument('-n', '--experiment_id', type=str, required=True)
    return parser.parse_args()

args = parse_args()
EXPERIMENT_ID = args.experiment_id
STATEMENT_TYPE = 1

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.read_csv('/home/devanshg27/semtabfact/csv/'+item.table_file).astype(str) # be sure to make your table data text only
        encoding = self.tokenizer(table=table,
                                  queries=item.question,
                                  # answer_coordinates=item.answer_coordinates,
                                  # answer_text=item.answer_text,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors="pt"
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        # Handle row truncation
        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]
        row_ids = encoding["token_type_ids"][:, token_types.index("row_ids")]
        last_row = row_ids.max().item()
        valid_indices = [i for i, x in enumerate(item.answer_coordinates) if x[0] < last_row]
        
        encoding = self.tokenizer(table=table,
                                  queries=item.question,
                                  answer_coordinates=[item.answer_coordinates[i] for i in valid_indices],
                                  answer_text=[item.answer_text[i] for i in valid_indices],
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors="pt"
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        encoding["table_size"] = list(item.position)
        encoding["table_size"][1] = last_row
        encoding["answer_coordinates"] = str([item.answer_coordinates[i] for i in valid_indices])
        return encoding
    def __len__(self):
       return len(self.data)

class ScorePreds:
    def __init__(self, keep_model=False):
        self.keep_model = keep_model
        self.best_model = None
        self.best_metrics = [0]

    def update_best_score(self, model, metrics):
        if metrics[0] <= self.best_metrics[0]:
            return False
        print(f'Exceeded the past model with {self.best_metrics[0]} with a score of {metrics[0]}', flush=True)
        if self.keep_model:
            print('Saved model!', flush=True)
            self.best_model = copy.deepcopy(model)
            self.best_model.to('cpu')
        print(metrics, flush=True)
        self.best_metrics = metrics
        return True

    def score(self, model, tn, fp, fn, tp):
        try:
            metrics = (
                tp/(tp + 0.5*(fp+fn)),
                [tn/(tn+fn), tp/(tp+fp)],
                [tn/(tn+fp), tp/(tp+fn)],
                (tp+tn)/(tn+fn+fp+tp) * 100,
                [[tn, fp], [fn, tp]],
                )
        except ZeroDivisionError:
            metrics = (0, 0)

        updated = self.update_best_score(model, metrics)
        return metrics, updated

device = 'cuda:0' if cuda.is_available() else 'cpu'
device

model_name = 'google/tapas-base-finetuned-wtq'
config = TapasConfig.from_pretrained(model_name)
config.num_aggregation_labels = 0
config.select_one_column = False
config.positive_label_weight = 10.0
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-tabfact', config=config)
model.to(device)
print('Model loaded!')

tokenizer = TapasTokenizer.from_pretrained(model_name)

def filter_data(df):
    if STATEMENT_TYPE is not None:
        df = df[df['annotator'] == STATEMENT_TYPE].reset_index(drop=True)
    df['answer_text'] = df['answer_text'].map(eval)
    df['answer_coordinates'] = df['answer_coordinates'].map(eval)
    df['position'] = df['position'].map(eval)
    return df

train_data = pd.read_csv('/home/devanshg27/semtabfact/tsv/train_auto_cell.tsv', sep='\t')
train_data = filter_data(train_data)

train_dataset = TableDataset(train_data, tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8)
fixed_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

test_data = pd.read_csv('/home/devanshg27/semtabfact/tsv/test_b_cell.tsv', sep='\t')
test_data = filter_data(test_data)

test_dataset = TableDataset(test_data, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

def validate(dataloader, fast=True, return_preds=False):
    tn, fp, fn, tp = 0, 0, 0, 0
    ret_preds = []

    model.eval()
    with torch.no_grad():
        for (_, batch_data) in tqdm(enumerate(dataloader, 0), disable=True):
            if fast and _ >= 10:
                break

            input_ids = batch_data['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch_data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch_data['token_type_ids'].to(device, dtype=torch.long)
            labels = batch_data["labels"].to(device, dtype=torch.float)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)

            predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(
                batch_data,
                outputs.logits.cpu().detach()
            )[0]

            if return_preds:
                ret_preds.extend(predicted_answer_coordinates)

            gt = batch_data['answer_coordinates']

            assert(len(predicted_answer_coordinates) == len(gt))

            for i in range(len(gt)):
                a_pred = set(predicted_answer_coordinates[i])
                b_gt = set(eval(gt[i]))
                tp += len(a_pred & b_gt)
                fp += len(a_pred - b_gt)
                fn += len(b_gt - a_pred)
                tn += ((batch_data['table_size'][1][i]-batch_data['table_size'][0][i]) * batch_data['table_size'][2][i]).item() - len(a_pred | b_gt)

    if return_preds:
        return tn, fp, fn, tp, ret_preds
    else:
        return tn, fp, fn, tp

dev_data = pd.read_csv('/home/devanshg27/semtabfact/tsv/dev_cell.tsv', sep='\t')
dev_data = filter_data(dev_data)

dev_dataset = TableDataset(dev_data, tokenizer)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=32)

main_scorer = ScorePreds(keep_model=True)

tn, fp, fn, tp = validate(dev_dataloader, fast=False, return_preds=False)
print(main_scorer.score(model, tn, fp, fn, tp)[0], flush=True)

fast_train_scorer = ScorePreds()
fast_val_scorer = ScorePreds()

def train(epochs=1, steps=0):
    if epochs == 0 and steps > 0:
        epochs = 1
    for epoch in range(epochs):
        model.train()
        # model.l1.eval()
        for idx, batch_data in enumerate(train_dataloader):
            # get the inputs;
            input_ids = batch_data["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch_data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = batch_data["token_type_ids"].to(device, dtype=torch.long)
            labels = batch_data["labels"].to(device, dtype=torch.float)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward   backward   optimize
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

            if idx%5 == 0:
                print(f"Step: {idx}, Epoch: {epoch}, Loss:  {loss.item()}", flush=True)

            loss.backward()
            optimizer.step()

            if steps > 0 and idx > steps:
                break

            if idx > 0 and idx%50 == 0:
                tn, fp, fn, tp, preds = validate(dev_dataloader, fast=False, return_preds=True)
                metrics, updated = main_scorer.score(model, tn, fp, fn, tp)
                print(metrics)
                if updated:
                    print(f'New record on val = {metrics}')
                    tn, fp, fn, tp, test_preds = validate(test_dataloader, fast=False, return_preds=True)
                    data = {
                        'dev': {
                            'preds': preds,
                            'metrics': metrics
                        },
                        'test': {
                            'preds': test_preds,
                        }
                    }
                    print(f'test set preds => {test_preds}')

                    save_model(model, f'tapas_{idx}', data)

                model.train()

    return epochs

def save_model(model, name, data):
    DIR = f'/scratch/devanshg27/models/table/{EXPERIMENT_ID}'
    model.save_pretrained(f'{DIR}/models/{name}/')
    config.save_pretrained(f'{DIR}/models/{name}/')
    with open(f'{DIR}/data/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Model stored!', flush=True)


TRAIN_EPOCHS = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print('\n\nTraining entire model...\n', flush=True)
for i in range(TRAIN_EPOCHS):
    train(1, steps=5000)

model = main_scorer.best_model.to(device)
save_model(model, 'tapas_best_model', {'dev': {'metrics': main_scorer.best_metrics}})