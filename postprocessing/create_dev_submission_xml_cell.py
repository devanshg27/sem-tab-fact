import sys
import pickle
with open(sys.argv[1], 'rb') as f:
	x = pickle.load(f)
	preds = x['dev']['preds']

print(len(preds))

from bs4 import BeautifulSoup
import glob
import collections
import shutil
import os

def remax(x, val):
	return val if (x is None or x < val) else x

dev_files = sorted(glob.glob('/home/devanshg27/semtabfact/xml/dev/v1.1/input/*.xml'))

SUBMISSION_DIR = './submit'

if os.path.exists(SUBMISSION_DIR) and os.path.isdir(SUBMISSION_DIR):
    shutil.rmtree(SUBMISSION_DIR)

os.makedirs(SUBMISSION_DIR)

idx = 0
for name in dev_files:
	with open(name) as f:
		soup = BeautifulSoup(f, 'xml')
		tables = soup.find_all("table")
		for table in tables:
			# Calculate no of rows and columns
			numRows = None
			numCols = None
			for row in table.find_all("row"):
				numRows = remax(numRows, int(row['row']))
				for cell in row.find_all("cell"):
					numCols = remax(numCols, int(cell['col-end']))
					numRows = remax(numRows, int(cell['row-end']))
			numRows += 1
			numCols += 1
			# Fill the table in arr
			statements_evidence_cnt = {}
			arr = [[None]*numCols for i in range(numRows)]
			for row in table.find_all("row"):
				for cell in row.find_all("cell"):
					for i in range(int(cell['row-start']), int(cell['row-end'])+1):
						for j in range(int(cell['col-start']), int(cell['col-end'])+1):
							assert(arr[i][j] is None)
							arr[i][j] = cell['text']
					# Evidence:
					for evidence in cell.find_all("evidence"):
						if evidence["statement_id"] not in statements_evidence_cnt:
							statements_evidence_cnt[evidence["statement_id"]] = 0
						statements_evidence_cnt[evidence["statement_id"]] += (1 + int(cell['row-end']) - int(cell['row-start'])) * (1 + int(cell['col-end']) - int(cell['col-start']))
			for i in range(numRows):
				for j in range(numCols):
					assert(arr[i][j] is not None)
			for cnt in statements_evidence_cnt.values():
				assert(cnt == numRows * numCols)
			# Calculate no. of header rows
			numHeaders = 1
			for i in range(1, numRows):
				if not(arr[i][0] == "" or arr[i][0] == arr[i-1][0]):
					numHeaders = remax(numHeaders, i)
					break
			while numHeaders < len(arr):
				increase_header_flag = False
				for j in range(1, numCols):
					if [arr[i][j-1] for i in range(numHeaders)] == [arr[i][j] for i in range(numHeaders)]:
						increase_header_flag = True
				if increase_header_flag:
					numHeaders += 1
				else:
					break
			if numHeaders == len(arr):
				numHeaders = 1
			# Merge header rows
			final_arr = [['\n'.join([arr[i][j] for i in range(numHeaders)]) for j in range(numCols)]]
			for i in range(numHeaders, numRows):
				final_arr.append(arr[i])
			# Read statements
			check_all_statements = 0
			for statement in table.find_all("statement"):
				statement['type'] = "entailed"
				if statement['id'] not in statements_evidence_cnt:
					continue
				check_all_statements += 1
				
			assert(check_all_statements == len(statements_evidence_cnt))
			cur_table_preds = [set(x) for x in preds[idx:idx+check_all_statements]]
			for i in range(len(cur_table_preds)):
				to_merge = set()
				for j in cur_table_preds[i]:
					to_merge.update([(i-numHeaders, j[1]) for i in range(numHeaders)])
				cur_table_preds[i] = (cur_table_preds[i] | to_merge)
			idx += check_all_statements
			for row in table.find_all("row"):
				for cell in row.find_all("cell"):
					cur_table_idx = -1
					for evidence in cell.find_all("evidence"):
						cur_table_idx += 1
						evidence["version"] = "0"
						evidence["type"] = "irrelevant"
						if cell['text'].strip() == "":
							continue
						for i in range(int(cell['row-start']), int(cell['row-end'])+1):
							for j in range(int(cell['col-start']), int(cell['col-end'])+1):
								if (i-numHeaders, j) in cur_table_preds[cur_table_idx]:
									evidence["type"] = "relevant"
		f2 = open(SUBMISSION_DIR + '/' + os.path.basename(name), "w")
		f2.write(soup.prettify())
		f2.close()

assert(idx == len(preds))

shutil.make_archive('submission', 'zip', SUBMISSION_DIR)