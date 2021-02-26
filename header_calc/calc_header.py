def remax(x, val):
	return val if (x is None or x < val) else x

class TableHelper:
	@staticmethod
	def calc_header(arr, numRows, numCols, table):
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
		return numHeaders