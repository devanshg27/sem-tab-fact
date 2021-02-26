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

model_name = 'google/tapas-base'
model = TapasForSequenceClassification.from_pretrained(model_name)
model.to(device)
model_name

def freeze(model):
    for name, p in model.named_parameters():
        p.requires_grad = False

def heat(model):
    for name, p in model.named_parameters():
        p.requires_grad = True

freeze(model)
hidden_size = model.config.hidden_size
if len(labels) != 2:
    model.classifier = torch.nn.Linear(hidden_size, len(labels))
heat(model.classifier)
model.classifier.to(device)
model.classifier

print(model, flush=True)

tokenizer = TapasTokenizer.from_pretrained(model_name)

def filter_data(df, labels):
    df = df[df['answer_text'].isin(labels)]
    return df

train_data = pd.read_csv(TSV_FOLDER+'train_3way_set.tsv', sep='\t')
train_data = filter_data(train_data, labels)
train_data

train_dataset = TableDataset(train_data, tokenizer)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=8)
fixed_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

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

optimizer = None

sizes = []
max_size = 0
for label in labels:
    size = len(train_data[train_data['answer_text'] == label])
    max_size = max(max_size, size)
    sizes.append(size)
w = max_size / np.asarray(sizes)
w = torch.from_numpy(w).float().to(device)
w = torch.pow(w, 1)
print(w, flush=True)

criterion = torch.nn.CrossEntropyLoss(weight=w)

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
            labels = batch_data["labels"].to(device, dtype=torch.long)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward   backward   optimize
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                # labels=labels,
            )
            loss = criterion(outputs.logits, labels)

            if idx%5 == 0:
                print(f"Epoch Step: {idx}, Loss:  {loss.item()}", flush=True)

            loss.backward()
            optimizer.step()

            if TESTING:
                break

            if steps > 0 and idx >= steps:
                break
            if idx > 0 and idx%100 == 0:
                print(main_scorer.calc_score(model, *validate(dev_dataloader, fast=False)), flush=True)
                model.train()
    return epochs

print(main_scorer.calc_score(model, *validate(dev_dataloader, fast=False)), flush=True)

CLASSIFIER_EPOCHS = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print('\n\nTraining classifier...\n', flush=True)
for i in range(CLASSIFIER_EPOCHS):
    train(1)
    print(f'Classifier Epochs = {i + 1}, Epochs = {0}', flush=True)
    preds, gt = validate(dev_dataloader, fast=False)
    metrics = main_scorer.calc_score(model, preds, gt)
    print('VAL SET SCORE', metrics, flush=True)

TRAIN_EPOCHS = 10

heat(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

print('\n\nTraining entire model...\n', flush=True)
for i in range(TRAIN_EPOCHS):
    train(1)
    print(f'Classifier Epochs = {CLASSIFIER_EPOCHS}, Epochs = {i + 1}', flush=True)
    preds, gt = validate(dev_dataloader, fast=False)
    metrics = main_scorer.calc_score(model, preds, gt)
    print('VAL SET SCORE', metrics, flush=True)
