import sys
ENTAIL_FILE = sys.argv[1]
REFUTE_FILE = sys.argv[2]


import pickle
import pandas as pd

with open(ENTAIL_FILE, 'rb') as f:
	x = pickle.load(f)
	dev_preds_entail = x['dev']['preds']
	test_preds_entail = x['test']['preds']

with open(REFUTE_FILE, 'rb') as f:
	x = pickle.load(f)
	dev_preds_refute = x['dev']['preds']
	test_preds_refute = x['test']['preds']

print('Dev: ', len(dev_preds_entail), len(dev_preds_refute), len(dev_preds_entail)+len(dev_preds_refute))
print('Test: ', len(test_preds_entail), len(test_preds_refute), len(test_preds_entail)+len(test_preds_refute))

merged_dev_preds = []
merged_test_preds = []


dev_data = pd.read_csv('/home/devanshg27/semtabfact/tsv/dev_cell.tsv', sep='\t')
assert(len(dev_preds_entail) + len(dev_preds_refute) == len(dev_data))

idx0 = 0
idx1 = 0

for i in range(len(dev_data)):
	if dev_data.iloc[i].annotator == 0:
		merged_dev_preds.append(dev_preds_refute[idx0])
		idx0 += 1
	elif dev_data.iloc[i].annotator == 1:
		merged_dev_preds.append(dev_preds_entail[idx1])
		idx1 += 1
	else:
		assert(False)

assert(idx0 == len(dev_preds_refute))
assert(idx1 == len(dev_preds_entail))


test_data = pd.read_csv('/home/devanshg27/semtabfact/tsv/test_b_cell.tsv', sep='\t')

assert(len(test_preds_entail) + len(test_preds_refute) == len(test_data))

idx0 = 0
idx1 = 0

for i in range(len(test_data)):
	if test_data.iloc[i].annotator == 0:
		merged_test_preds.append(test_preds_refute[idx0])
		idx0 += 1
	elif test_data.iloc[i].annotator == 1:
		merged_test_preds.append(test_preds_entail[idx1])
		idx1 += 1
	else:
		assert(False)

assert(idx0 == len(test_preds_refute))
assert(idx1 == len(test_preds_entail))

data = {
	'dev': {
		'preds': merged_dev_preds,
	},
	'test': {
		'preds': merged_test_preds,
	}
}

with open('/home/devanshg27/delete/temp.pkl', 'wb') as f:
	pickle.dump(data, f)