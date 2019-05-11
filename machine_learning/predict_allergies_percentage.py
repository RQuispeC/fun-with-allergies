from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os.path as osp
import gc


def get_model(x, y):
	regr = SVR()
	pipe = Pipeline(steps=[('reg', regr)])
	param_grid = {
		'reg__kernel':('linear', 'rbf'),
		'reg__C': [0.01, 0.1, 1, 10],
		'reg__epsilon': [0.1, 0.2, 0.4, 0.5, 0.8, 1., 1.5, 2, 3],
		'reg__gamma': ['auto', 'scale'],
	}
	search = GridSearchCV(pipe, param_grid, iid=False, cv=5,
		return_train_score=False, n_jobs = 4)
	search.fit(x, y)
	return search.best_estimator_

def read(file_name):
	data = pd.read_csv(file_name, sep = '\t')
	x = np.array([[float(year)] for year in list(data)])
	y = np.array([[year] for year in np.array(data).reshape(-1)]).reshape(-1, )
	return x, y

if __name__ == '__main__':
	root = '../data/machine_learning'
	file_names = ['black_african_american.tsv', 'female.tsv', 'hispanic_latino.tsv', 'male.tsv', 'under_18_years.tsv', 'white.tsv']
	names = ['black african american', 'female ', 'hispanic latino', 'male', 'under 18 years', 'white']

	query = np.array([[2018], [2019], [2020]]).reshape(-1, )
	for fn, n in zip(file_names, names):
		x, y = read(osp.join(root, fn))
		
		#predict(x, y, np.array([[2018], [2020]]))
		model = get_model(x, y)
		y_model = model.predict(x)
		#y_query = model.predict(query)

		fig =plt.figure()
		plt.title(n)
		plt.scatter(x, y,  color='green')
		#plt.scatter(query, y_query,  color='black')
		plt.plot(x, y_model, color='blue', linewidth=2)

		plt.savefig(fn.split('.')[0] + '.jpg')

		# Clean RAM
		fig.clf()
		plt.close()
		gc.collect()