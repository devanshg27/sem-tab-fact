import torch
import pandas as pd
from transformers import TapasTokenizer, TapasForSequenceClassification
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

CSV_FOLDER = '/home/devanshg27/semtabfact/csv_noheader/'
TSV_FOLDER = '/home/devanshg27/semtabfact/tsv_noheader/'

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune TAPAS')
    parser.add_argument('-n', '--experiment_id', type=str, required=True)
    return parser.parse_args()

args = parse_args()
EXPERIMENT_ID = args.experiment_id
TESTING=False

CONST_DICT = {'refuted': 0, 'entailed': 1, 'unknown': 2}

skip_labels = {}
# skip_labels = {'unknown': 2}

labels = []
label_names = []
for label in CONST_DICT.keys():
    if label not in skip_labels:
        label_names.append(label)
        labels.append(CONST_DICT[label])

label_names, labels

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        table = pd.read_csv(CSV_FOLDER+item.table_file).astype(str) # be sure to make your table data text only
        encoding = self.tokenizer(table=table,
                                  queries=item.question,
                                #   answer_coordinates=item.answer_coordinates,
                                #   answer_text=item.answer_text,
                                  truncation=True,
                                  padding="max_length",
                                  return_tensors="pt"
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        # add the float_answer which is also required (weak supervision for aggregation case)
        # encoding["float_answer"] = torch.tensor(item.float_answer)
        # print(item.answer_text, flush=True)
        encoding["labels"] = int(item.answer_text)
        return encoding
    def __len__(self):
       return len(self.data)

device = 'cuda:0' if cuda.is_available() else 'cpu'
device

model_name = 'google/tapas-base-finetuned-tabfact'
model = TapasForSequenceClassification.from_pretrained(model_name)
model.to(device)
model_name

print(model, flush=True)

tokenizer = TapasTokenizer.from_pretrained(model_name)

def filter_data(df, labels):
    df = df[df['answer_text'].isin(labels)]
    return df

test_data = pd.read_csv(TSV_FOLDER+'test_a_set.tsv', sep='\t')
test_data = filter_data(test_data, labels)
test_data

test_dataset = TableDataset(test_data, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

def validate(dataloader, fast=True):
    preds = []
    gt = []

    model.eval()
    with torch.no_grad():
        for (_, batch_data) in tqdm(enumerate(dataloader, 0), disable=True):
            input_ids = batch_data['input_ids'].to(device,
                    dtype=torch.long)
            attention_mask = batch_data['attention_mask'].to(device,
                    dtype=torch.long)
            token_type_ids = batch_data['token_type_ids'].to(device,
                    dtype=torch.long)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            temp = outputs.logits.cpu().detach().numpy().tolist()
            pred = [max(enumerate(x), key=lambda x: x[1])[0] for x in temp]
            preds.extend(pred)

            labels = batch_data['labels']
            gt.extend(labels)

            if fast and _ > 20:
                break

            if TESTING:
                break
    return preds, gt

dev_data = pd.read_csv(TSV_FOLDER+'dev_set.tsv', sep='\t')
dev_data = filter_data(dev_data, labels)
dev_data

dev_dataset = TableDataset(dev_data, tokenizer)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=32)

def save_model(model, name, data):
    DIR = f'/scratch/devanshg27/models/table/{EXPERIMENT_ID}'
    torch.save(model, f'{DIR}/models/{name}.bin')
    with open(f'{DIR}/data/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('Model stored!', flush=True)

class ScorePreds:
    def __init__(self, keep_model=False):
        self.keep_model = keep_model
        self.best_model = None
        self.best_metrics = [0]
        self.best_model_cnt = 0

    def update_best_score(self, model, metrics, preds):
        if metrics[0] <= self.best_metrics[0]:
            return
        print(f'Exceeded the past model with {self.best_metrics[0]} with a score of {metrics[0]}', flush=True)
        if self.keep_model:
            assert(False)
            # print('Saved model!', flush=True)
            # self.best_model = copy.deepcopy(model)
            # self.best_model.to('cpu')
        print(metrics, flush=True)
        test_preds, test_gt = validate(test_dataloader, fast=False)
        data = {
            'dev': {
                'preds': preds,
                'metrics': metrics
            },
            'test': {
                'preds': test_preds,
            }
        }
        save_model(model, f'tapas_tabfact_{self.best_model_cnt}', data)
        self.best_model_cnt += 1
        self.best_metrics = metrics

    def calc_score(self, model, preds, gt):
        assert len(preds) == len(gt)
        
        if len(labels) == 2:
            assert(False)
        else:
            metrics = (
                f1_score(gt, preds, average='micro', labels=labels),
                f1_score(gt, preds, average='macro', labels=labels),
                accuracy_score(gt, preds) * 100,
                confusion_matrix(gt, preds),
            )

        self.update_best_score(model, metrics, preds)
        return metrics

main_scorer = ScorePreds(keep_model=False)
print(main_scorer.calc_score(model, *validate(dev_dataloader, fast=False)), flush=True)