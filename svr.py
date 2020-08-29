import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []
windowlen = 10


def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[1]))
			prices.append(float(row[2]))

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_rbf = SVR(kernel= 'rbf', C= 150, gamma= 0.01) # defining the support vector regression models
	svr_rbf.fit(dates, prices) # fitting the data points in the models

	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(np.array(x).reshape(-1,1))[0]

get_data('50corn.csv')
print("Dates- ", dates)
print("Prices- ", prices)

predicted_price = predict_price(dates, prices, (len(prices) + windowlen))

print("\nThe window length is: " + str(windowlen))
print("\nThe stock open price is: " + str(predicted_price))