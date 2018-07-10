import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class NeuralNetwork:

	def __init__(self, inputLen, hiddenLayers = []):
		tf.reset_default_graph()
		self.winit = tf.glorot_uniform_initializer()
		self.binit = tf.constant_initializer(0.0)

		self.hiddenLayers = hiddenLayers

		self.inputs = tf.placeholder(tf.float64, [None, inputLen])
		self.labels = tf.placeholder(tf.float64, [None, 1])
		self.batchSize = tf.placeholder(tf.float64, [])

		self.learningRate = tf.placeholder_with_default(np.float64(0.01),  [])
		self.regularization = tf.placeholder_with_default(np.float64(0.1),  [])
		self.keepProb = tf.placeholder_with_default(np.float64(1.0),  [])

		lastOutputs = tf.nn.dropout(self.inputs, self.keepProb)
		lastSize = inputLen
		i = 0
		weigths = []
		for l in hiddenLayers:
			scope = "hidden_" + str(i)
			with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
				w = tf.get_variable(
					name = "w",
					shape = [lastSize, l],
					dtype = tf.float64,
					initializer = self.winit)
				b = tf.get_variable(
					name = "b",
					shape = [l],
					dtype = tf.float64,
					initializer = self.binit)
				#lastOutputs = tf.sigmoid(tf.matmul(lastOutputs, w) + b)
				lastOutputs = tf.nn.relu(tf.matmul(lastOutputs, w) + b)
				lastOutputs = tf.nn.dropout(lastOutputs, self.keepProb)
				lastSize = l
				weigths.append(w)
			i += 1
		with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
			w = tf.get_variable(
				name = "w",
				shape = [lastSize, 1],
				dtype = tf.float64,
				initializer = self.winit)
			b = tf.get_variable(
				name = "b",
				shape = [1],
				dtype = tf.float64,
				initializer = self.binit)
			weigths.append(w)

		self.predictions = tf.sigmoid(tf.matmul(lastOutputs, w) + b)
		self.cost = tf.reduce_mean(tf.losses.log_loss(self.labels, self.predictions))
		self.squaredWeigth = tf.add_n([tf.reduce_sum(tf.square(w)) for w in weigths])
		self.ncost = tf.cast(self.cost, tf.float64) + (self.regularization / (self.batchSize * 2.0)) * self.squaredWeigth 

		self.optim = tf.train.AdamOptimizer(self.learningRate).minimize(self.ncost)


		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		self.session.run(tf.local_variables_initializer())
	
	def reinit(self):
		self.session.run(tf.global_variables_initializer())
		self.session.run(tf.local_variables_initializer())

	def train(self, data, epoch, regularization = 0.01, learningRate = 0.01, keepProb = 1.0):
		cost = 0
		batchInput, batchOutput = data.trainBatch()
		for i in tqdm(range(epoch)):
			_, c = self.session.run([self.optim, self.ncost], feed_dict={
				self.batchSize: len(batchOutput),
				self.inputs: batchInput,
				self.labels: batchOutput,
				self.learningRate:learningRate,
				self.regularization:regularization,
				self.keepProb:keepProb
			})
			cost += c
		cost = cost / epoch
		print("cost : " + str(cost))
	
	def getCost(self, data):
		batchInput, batchOutput = data.trainBatch()
		trainCost = self.session.run(self.cost, feed_dict={
			self.batchSize: len(batchOutput),
			self.inputs: batchInput,
			self.labels: batchOutput,
			self.regularization:0,
			self.keepProb:1.0
		})
		batchInput, batchOutput = data.testBatch(100)
		testCost = self.session.run(self.cost, feed_dict={
			self.batchSize: len(batchOutput),
			self.inputs: batchInput,
			self.labels: batchOutput,
			self.regularization:0,
			self.keepProb:1.0
		})
		return trainCost, testCost

	def plotCostByReg(self, data, epoch = 5000, learningRate = 0.01, minReg = 0.00, maxReg = 1.0, step = 2):
		trainCost = []
		testCost = []
		xticks = []
		reg = minReg
		while reg < maxReg:
			print("reg " + str(reg) + "/" + str(maxReg))
			self.reinit()
			self.train(data, epoch, reg, learningRate, 1.0)
			trainC, testC = self.getCost(data)
			trainCost.append(trainC)
			testCost.append(testC)
			xticks.append(reg)
			reg += step
		self.plotLines([trainCost, testCost], ["train cost", "test cost"], xticks)
	
	def roc_auc_score(self, labels, predictions):
		return roc_auc_score(
			np.reshape(labels, [len(labels)]),
			np.reshape(predictions, [len(labels)])
		)

	def stats(self, predictions, labels, limit = 0.5):
		predictions[predictions < limit] = 0
		predictions[predictions >= limit] = 1
		truePositives = len(predictions[(predictions == labels) & (labels == 1)])
		trueNegatives = len(predictions[(predictions == labels) & (labels == 0)])
		falsePositives = len(predictions[(predictions != labels) & (predictions == 1)])
		falseNegatives = len(predictions[(predictions != labels) & (predictions == 0)])

		precision = 0
		recall = 0
		f1 = 0
	
		if truePositives > 0:
			precision = truePositives / (truePositives+falsePositives)
			recall = truePositives / (truePositives+falseNegatives)
			f1 = 2 * (recall*precision) / (recall+precision)

		return precision,recall,f1
	

	def printStats(self, predictions, labels, limit = 0.5):
		precision,recall,f1 = self.stats(predictions, labels, limit)
		print("precision : " + str(precision) + " (" + str(int(precision * 100)) + "%)")
		print("recall : " + str(recall) + " (" + str(int(recall * 100)) + "%)")
		print("f1 score : " + str(f1))

	def check(self, data, limit = 0.5, regularization = 0.0, keepProb = 1.0):
		batchInput, batchOutput = data.testBatch()
		p, cost = self.session.run([self.predictions, self.ncost], feed_dict={
			self.batchSize: len(batchOutput),
			self.inputs: batchInput,
			self.labels: batchOutput,
			self.regularization: regularization,
			self.keepProb: keepProb
		})
		metric = self.roc_auc_score(batchOutput[:], p[:])
		self.printStats(p[:], batchOutput[:], limit)
		print("cost : " + str(cost))
		#print("metric : " + str(metric))
	
	def plotLines(self, arrays, labels, xticks):
		fig = plt.figure(figsize=(15, 8))
		lines = []
		for i in range(len(arrays)):
			lines.append(plt.plot(arrays[i], label=labels[i]))

		plt.xticks(np.arange(len(xticks)), xticks)
		ax = plt.gca()
		ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.show()
	
	def statsPlot(self, data, regularization = 0.0, keepProb = 1.0):
		f1s = []
		precisions = []
		recalls = []
		xticks = []

		batchInput, batchOutput = data.testBatch()
		predictions, cost = self.session.run([self.predictions, self.ncost], feed_dict={
			self.batchSize: len(batchOutput),
			self.inputs: batchInput,
			self.labels: batchOutput,
			self.regularization: regularization,
			self.keepProb: keepProb
		})
		for limit in np.arange(0.01, 1.0, 0.02):
			p,r,f = self.stats(np.copy(predictions), np.copy(batchOutput), limit=limit)
			f1s.append(f)
			precisions.append(p)
			recalls.append(r)
			xticks.append(limit)
		self.plotLines([precisions, recalls, f1s], ["precision", "recall", "f1 score"], xticks)

	def tsnePlot(self, data, limit = 0.5, regularization = 0.0, keepProb = 1.0):
		batchInput, batchOutput = data.testBatch()
		predictions, cost = self.session.run([self.predictions, self.ncost], feed_dict={
			self.batchSize: len(batchOutput),
			self.inputs: batchInput,
			self.labels: batchOutput,
			self.regularization: regularization,
			self.keepProb: keepProb
		})
		predictions[predictions < limit] = 0
		predictions[predictions >= limit] = 1
		tsneData = TSNE(n_components=2).fit_transform(batchInput)
		fig = plt.figure(figsize=(10, 8))
		i = 0
		for d in tqdm(tsneData):
			x = d[0]
			y = d[1]
			if batchOutput[i][0] == 0:
				if predictions[i][0] == batchOutput[i][0]:
					plt.plot(x, y, 'gx')
				else:
					plt.plot(x, y, 'rx')
			else:
				if predictions[i][0] == batchOutput[i][0]:
					plt.plot(x, y, 'go')
				else:
					plt.plot(x, y, 'ro')
			i += 1
		plt.show()
