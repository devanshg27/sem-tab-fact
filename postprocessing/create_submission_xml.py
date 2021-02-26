import sys
import pickle
with open(sys.argv[1], 'rb') as f:
	x = pickle.load(f)
	preds = x['test']['preds']

from bs4 import BeautifulSoup
import glob
import collections
import shutil
import os

dev_files = sorted(glob.glob('/home/devanshg27/semtabfact/xml/test_a/*.xml'))

SUBMISSION_DIR = './submit'

if os.path.exists(SUBMISSION_DIR) and os.path.isdir(SUBMISSION_DIR):
    shutil.rmtree(SUBMISSION_DIR)

os.makedirs(SUBMISSION_DIR)

CONST_DICT = {
	'refuted': 0,
	'entailed': 1,
	'unknown': 2,
}

REV_CONST_DICT = {
	0: 'refuted',
	1: 'entailed',
	2: 'unknown',
}

idx = 0
for name in dev_files:
	with open(name) as f:
		soup = BeautifulSoup(f, 'xml')
		statements = soup.find_all("statement")

		for statement in statements:
			statement['type'] = REV_CONST_DICT[int(preds[idx])]
			idx += 1

		f2 = open(SUBMISSION_DIR + '/' + os.path.basename(name), "w")
		f2.write(soup.prettify())
		f2.close()

assert(idx == len(preds))

shutil.make_archive('submission', 'zip', SUBMISSION_DIR)