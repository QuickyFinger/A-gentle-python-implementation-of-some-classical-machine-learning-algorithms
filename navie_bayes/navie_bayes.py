# bayes classifier using ayes estimation
import pandas as pd

class bayes(object):
	def __init__(self, estimator=None, lamb = 1):
		if estimator == None:
			raise ValueError('estimator should be mle or be !!!')
		self.estimator = estimator
		self.lamb = lamb
		self.prob = dict()
		self.feature_names = None
		self.label_names = None

	# def maximum_likely_hood_estimator(self):

	def fit(self, x_data, y_data):

		data = pd.concat([x_data, y_data], axis=1)
		total_sample = len(data)
		total_cat = len(list(y_data.unique()))
		names = list(data.columns)

		self.feature_names = names[:-1]
		self.label_names = list(y_data.unique())
		# compute categorical probability

		for key, value in y_data.value_counts().items():
			self.prob[key] = value


		for feature in names[:-1]:
			num_feature = len(data[feature].unique())
			tmp = data.groupby([feature, names[-1]], as_index=False)[names[-1]].agg({'count':'count'})
			for key, label, value in tmp.items():
				self.prob[(feature, key, label)] = (value + self.lamb) / (self.prob[label] + num_feature * self.lamb)

		for key in list(y_data.unique()):
			self.prob[key] = (self.prob[key] + self.lamb) / (total_sample + total_cat * self.lamb)



	def predict(self, x_data):
		x_data['label'] = None
		# predict label 
		for index in range(len(x_data)):
			max_prob = 0
			label = None
			for cate in self.label_names:
				prob = self.prob[cate]
				for key, value in x_data.iloc[index].items():
					prob *= self.prob[(key, value, cate)]
				if prob >= max_prob:
					max_prob = prob
					label = cate
			x_data.iloc[index]['label'] = label

		return x_data['label']
