from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def predict(x, y, queries):
	#regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=20)
	regr = LinearRegression()
	regr.fit(x, y)
	pred = regr.predict(queries)
	for i, j in zip(queries, pred):
		print("In ",i,"there will be", j,"percentage of people with allergy")

def read():
	data = pd.read_csv('toy_data.csv', sep = '\t')
	x = np.array([[float(year)] for year in list(data)])
	y = np.array([[year] for year in np.array(data).reshape(-1)])
	return x, y

if __name__ == '__main__':
	x, y = read()
	predict(x, y, np.array([[2018], [2020]]))

#TODO, add plot https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
