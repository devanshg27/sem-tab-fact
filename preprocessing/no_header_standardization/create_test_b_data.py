import glob
import csv
from bs4 import BeautifulSoup
from tqdm import tqdm

test_b_files = sorted(glob.glob('/home/devanshg27/semtabfact/xml/test_b/*.xml'))

print(len(test_b_files))


CSV_FOLDER = '/home/devanshg27/semtabfact/csv_noheader/'
TSV_FOLDER = '/home/devanshg27/semtabfact/tsv_noheader/'

def remax(x, val):
	return val if (x is None or x < val) else x

header = ('id', 'annotator', 'position', 'question', 'table_file',
			'answer_coordinates', 'answer_text', 'aggregation', 'float_answer')

CONST_DICT = {
	'refuted': 0,
	'entailed': 1,
	'unknown': 2,
}

def read_xml(f, table_idx, tsv_writer, file_prefix):
	soup = BeautifulSoup(f, 'xml')
	tables = soup.find_all("table")
	# Read Tables
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
		for i in range(numRows):
			for j in range(numCols):
				assert(arr[i][j] is not None)
		# Calculate no. of header rows
		numHeaders = 1
		# for i in range(1, numRows):
		# 	if not(arr[i][0] == "" or arr[i][0] == arr[i-1][0]):
		# 		numHeaders = remax(numHeaders, i)
		# 		break
		# while numHeaders < len(arr):
		# 	increase_header_flag = False
		# 	for j in range(1, numCols):
		# 		if [arr[i][j-1] for i in range(numHeaders)] == [arr[i][j] for i in range(numHeaders)]:
		# 			increase_header_flag = True
		# 	if increase_header_flag:
		# 		numHeaders += 1
		# 	else:
		# 		break
		# if numHeaders == len(arr):
		# 	numHeaders = 1
		# Merge header rows
		final_arr = [['\n'.join([arr[i][j] for i in range(numHeaders)]) for j in range(numCols)]]
		for i in range(numHeaders, numRows):
			final_arr.append(arr[i])
		# Read statements
		check_all_statements = 0
		for statement in table.find_all("statement"):
			if statement['id'] not in statements_evidence_cnt:
				continue
			assert(statement["type"] != "unknown")
			check_all_statements += 1
			tsv_writer.writerow({
				'id':
					table_idx,
				'annotator':
					CONST_DICT[statement["type"]],
				'position':
					(numHeaders, numRows, numCols),
				'question':
					statement['text'],
				'table_file':
					f"{file_prefix}{table_idx}.csv",
				'answer_coordinates':
					"[]",
				'answer_text':
					"[]",
				'aggregation':
					None,
				'float_answer':
					None,
			})
		assert(check_all_statements == len(statements_evidence_cnt))
		# Write the table
		with open(CSV_FOLDER + file_prefix + str(table_idx) + '.csv', 'w') as csvfile:
			writer = csv.writer(csvfile)
			for i in range(len(final_arr)):
				writer.writerow(final_arr[i])
		table_idx += 1
	return table_idx

with open(TSV_FOLDER + 'test_b_cell.tsv', 'w') as tsvfile:
	tsv_writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=header)
	tsv_writer.writeheader()
	table_idx = 1
	for file in tqdm(test_b_files):
		with open(file) as f:
			table_idx = read_xml(f, table_idx, tsv_writer, 'test_b_')
