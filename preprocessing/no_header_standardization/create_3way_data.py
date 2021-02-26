import csv
from tqdm import tqdm
import random

header = ('id', 'annotator', 'position', 'question', 'table_file',
			'answer_coordinates', 'answer_text', 'aggregation', 'float_answer')

TSV_FOLDER = '/home/devanshg27/semtabfact/tsv_noheader/'

statements = []

with open(TSV_FOLDER + 'train_man_set.tsv', 'r') as infile:
	input_file = csv.DictReader(infile, delimiter='\t')
	statements = [(row['id'], row['question'], row['table_file'], row['answer_text']) for row in input_file]


tables = {}

for s in statements:
	tables[int(s[0])] = s[2]

with open(TSV_FOLDER + 'train_3way_set.tsv', 'w') as tsvfile:
	tsv_writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=header)
	tsv_writer.writeheader()
	for s in tqdm(statements):
		tsv_writer.writerow({
			'id':
				s[0],
			'annotator':
				'0',
			'position':
				'0',
			'question':
				s[1],
			'table_file':
				s[2],
			'answer_coordinates': [],
			'answer_text':
				s[3],
			'aggregation':
				None,
			'float_answer':
				None,
		})
	for s in tqdm(statements):
		other_table = random.randint(1, len(tables)-1)
		if other_table >= int(s[0]):
			other_table += 1
		tsv_writer.writerow({
			'id':
				other_table,
			'annotator':
				'0',
			'position':
				'0',
			'question':
				s[1],
			'table_file':
				tables[other_table],
			'answer_coordinates': [],
			'answer_text':
				2,
			'aggregation':
				None,
			'float_answer':
				None,
		})