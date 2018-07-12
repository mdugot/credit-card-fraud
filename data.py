from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

class Data:

	def __init__(self):
		delimiter = ","
		print("load data")
		with open("creditcard.csv") as file:
			lines = file.readlines()
			names = lines[0].strip().split(delimiter)
			self.data = np.zeros([len(lines)-1, len(names)-1], dtype=np.float64)
			self.output = np.zeros([len(lines)-1, 1], dtype=np.float64)
			
			for i in tqdm(range(len(lines))):
				if i > 0 :
					line = lines[i]
					tmp = np.fromstring(line, sep=delimiter, dtype=np.float64)
					self.data[i - 1] = tmp
					self.output[i - 1] = np.float64(line.split(delimiter)[-1].strip('"\n'))
		means = np.mean(self.data, 0)
		ranges = np.max(self.data, 0) - np.min(self.data, 0)
		self.data = (self.data - means) / ranges
		self.trueIndex = np.where(self.output == 1)[0]
		self.falseIndex = np.where(self.output == 0)[0]

		self.trueIndex_train = np.random.choice(self.trueIndex, 400, replace=False)
		self.trueIndex_test = np.array([i for i in self.trueIndex if i not in self.trueIndex_train])

		self.falseIndex_train = np.random.choice(self.falseIndex, 400, replace=False)
		self.falseIndex_test = np.array([i for i in self.falseIndex if i not in self.falseIndex_train])

	def trainBatch(self):
		batchIndex = np.append(self.trueIndex_train, self.falseIndex_train)
		np.random.shuffle(batchIndex)
		return self.data[batchIndex], self.output[batchIndex]

	def testBatch(self, sizeFalse = 1000):
		falseTestIndexSample = np.random.choice(self.falseIndex_test, 1000, replace=False)
		batchIndex = np.append(self.trueIndex_test, falseTestIndexSample)
		np.random.shuffle(batchIndex)
		return self.data[batchIndex], self.output[batchIndex]


	def sampleData(self, nfalse):
		falseSample = np.random.choice(self.falseIndex, nfalse, replace=False)
		sample = np.append(self.trueIndex, falseSample)
		return self.data[sample]

	def tsnePlot(self):
		tsneData = TSNE(n_components=2).fit_transform(self.sampleData(1000))
		fig = plt.figure(figsize=(10, 8))
		i = 0
		for d in tqdm(tsneData):
			x = d[0]
			y = d[1]
			if i < len(self.trueIndex):
				plt.plot(x, y, 'ro')
			else:
				plt.plot(x, y, 'bo')
			i += 1
		bo = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='normal transaction')
		ro = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='fraud transaction')
		plt.legend(handles=[ro, bo])
		plt.title("Data visualization with t-SNE")
		plt.show()
