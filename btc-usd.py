import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from orders import *
from coin_hist_pull import *
from g_trends import g_trend_pull

import warnings
warnings.filterwarnings('ignore')

def get_btc_hist_tms(coin):
	hist = pd.read_csv("data/" + coin +  "_btc.csv")
	hist = hist.iloc[::-1]
	return hist

# Create variables for machine learning model
def hist_to_model_prep(coin_hist):
	coin_hist["prev"] = coin_hist["btc_price"].shift(periods=1)
	coin_hist["btc_price2"] = coin_hist["btc_price"].rolling(2, win_type='triang').mean()
	coin_hist["btc_price3"] = coin_hist["btc_price"].rolling(3, win_type='triang').mean()
	coin_hist["btc_price4"] = coin_hist["btc_price"].rolling(4, win_type='triang').mean()
	coin_hist["btc_price5"] = coin_hist["btc_price"].rolling(5, win_type='triang').mean()
	coin_hist["btc_price6"] = coin_hist["btc_price"].rolling(6, win_type='triang').mean()
	coin_hist["btc_price7"] = coin_hist["btc_price"].rolling(7, win_type='triang').mean()
	coin_hist["btc_price8"] = coin_hist["btc_price"].rolling(8, win_type='triang').mean()
	coin_hist["btc_price9"] = coin_hist["btc_price"].rolling(9, win_type='triang').mean()
	coin_hist["btc_price10"] = coin_hist["btc_price"].rolling(10, win_type='triang').mean()
	coin_hist["btc_price11"] = coin_hist["btc_price"].rolling(11, win_type='triang').mean()
	coin_hist["btc_price12"] = coin_hist["btc_price"].rolling(12, win_type='triang').mean()
	coin_hist["btc_price13"] = coin_hist["btc_price"].rolling(13, win_type='triang').mean()
	coin_hist["btc_price14"] = coin_hist["btc_price"].rolling(14, win_type='triang').mean()
	coin_hist["btc_price15"] = coin_hist["btc_price"].rolling(15, win_type='triang').mean()
	coin_hist["btc_price16"] = coin_hist["btc_price"].rolling(16, win_type='triang').mean()
	coin_hist["btc_price17"] = coin_hist["btc_price"].rolling(17, win_type='triang').mean()
	coin_hist["btc_price18"] = coin_hist["btc_price"].rolling(18, win_type='triang').mean()
	coin_hist["btc_price19"] = coin_hist["btc_price"].rolling(19, win_type='triang').mean()
	coin_hist["btc_price20"] = coin_hist["btc_price"].rolling(20, win_type='triang').mean()
	coin_hist["btc_price24"] = coin_hist["btc_price"].rolling(24, win_type='triang').mean()
	coin_hist["btc_price72"] = coin_hist["btc_price"].rolling(72, win_type='triang').mean()
	coin_hist["next"] = coin_hist["btc_price12"].shift(periods=-24)
	coin_hist["n_bp"] = (coin_hist["next"]/coin_hist["btc_price"] - 1)
	coin_hist["sell"] = coin_hist["n_bp"] > np.mean(coin_hist["n_bp"]) + 1*np.std(coin_hist["n_bp"])
	coin_hist["sell"] = coin_hist["sell"].astype('int32')
	coin_hist["go"] = coin_hist["n_bp"] < np.mean(coin_hist["n_bp"]) - 1*np.std(coin_hist["n_bp"])
	coin_hist["go"] = coin_hist["go"].astype('int32')

	coin_hist["p_go"] = 0
	coin_hist["p_sell"] = 0
	coin_hist["o3_go"] =  0
	coin_hist["o3_sell"] =  0
	coin_hist["conviction_go"] = 0
	coin_hist["conviction_sell"] = 0
	coin_hist["sell_signal"] =  0
	coin_hist["go_signal"] = 0

	g_trends = pd.DataFrame(g_trend_pull(index))
	coin_hist['dt'] = pd.to_datetime(coin_hist['market_tms'])
	coin_hist['dt'] = coin_hist['dt'].apply(lambda x: x.strftime("%Y-%m-%d"))
	g_trends['dt'] = g_trends.index
	g_trends['dt'] = g_trends['dt'].apply(lambda x: x.strftime("%Y-%m-%d"))
	coin_hist = pd.merge(coin_hist, g_trends, how="left", on=["dt"])

	coin_hist = coin_hist.fillna(method='ffill')
	coin_hist = coin_hist.dropna()

	return coin_hist

# Model prep and run
def model_build_and_run(coin_hist):
	ft_col = ['btc_price', 'btc_price2', 'btc_price3', 'btc_price4', 'btc_price5', 'btc_price6',
	'btc_price7', 'btc_price8', 'btc_price9', 'btc_price10', 'btc_price11', 'btc_price12', 'btc_price13',
	'btc_price14', 'btc_price15', 'btc_price16', 'btc_price17', 'btc_price18', 'btc_price19', 'btc_price20',
	'btc_price24', 'btc_price72', 'bitcoin']

	X = coin_hist[ft_col]
	output = coin_hist[ft_col].tail(1)
	coin_hist.drop(coin_hist.tail(1).index, inplace=True)

	scaler = StandardScaler().fit(X)
	red = scaler.transform(X)
	red = pd.DataFrame(red)
	y = red.tail(1)
	for column in red:
		col = red[column]
		col_name = red[column].name
		if col_name != "go" or col_name != "go":
			new_name = "Scale_" + str(col_name)
			red = red.rename(columns={col_name: new_name})

	red.drop(red.tail(1).index, inplace=True)

	# Go
	red['go'] = np.array(coin_hist['go'])
	x = red[red.columns[red.columns != 'go']]
	X_train, X_test, y_train, y_test = train_test_split(x, red['go'], test_size=0.25, random_state=0)
	go_build = autoML(X_train, X_test, y_train, y_test)
	p_go = pd.DataFrame(go_build.predict_proba(x)).iloc[:,0]
	p_go = np.array(p_go)
	o3_go = go_build.predict_proba(y)[[0]][0][0]

	# sell
	red = red.drop(columns=['go'])
	red['sell'] = np.array(coin_hist['sell'])
	x = red[red.columns[red.columns != 'sell']]
	X_train, X_test, y_train, y_test = train_test_split(x, red['sell'], test_size=0.25, random_state=0)
	sell_build = autoML(X_train, X_test, y_train, y_test)
	p_sell = pd.DataFrame(sell_build.predict_proba(x)).iloc[:,0]
	p_sell = np.array(p_sell)
	o3_sell = sell_build.predict_proba(y)[[0]][0][0]

	pgo_mean = np.mean(p_go)
	pgo_sd = np.std(p_go)
	psell_mean = np.mean(p_sell)
	psell_sd = np.std(p_sell)

	output["p_go"] = pgo_mean
	output["p_sell"] = psell_mean
	output["o3_go"] = o3_go
	output["o3_sell"] = o3_sell
	output["conviction_go"] = abs(o3_go/pgo_sd)/5
	output["conviction_sell"] = abs(o3_sell/psell_sd)/5
	output["go_signal"] =  int(o3_go - o3_sell > 0.12)
	output["sell_signal"] = int(o3_sell - o3_go > 0.12)

	# print(output)
	return output


def build_models_on_train( model, X_train,  y_train):
    classifier = model()
    classifier.fit(X_train, y_train)
    return classifier

def autoML(X_train, X_test, y_train, y_test):
	models_to_run = [LogisticRegression] #, SVR, MLPRegressor, GaussianNB, SGDRegressor]

	max_score = 0
	max_build = 0

	for algo in models_to_run:
		build = build_models_on_train(algo, X_train, y_train)
		pred = build.predict(X_test)
		# print("MAE = {:5.4f}".format(metrics.mean_absolute_error(y_test, pred)))
		# print("MSE = {:5.4f}".format(metrics.mean_squared_error(y_test, pred)))
		# print("RMSE = {:5.4f}".format(np.sqrt(metrics.mean_squared_error(y_test, pred))))
		# print("Score:", build.score(X_test, y_test))
		# print()

		if build.score(X_test, y_test) > max_score:
			max_score = build.score(X_test, y_test)
			max_build = build

	predicted = max_build.predict(X_test)

	# print()
	# print("RESULTS: ")
	# print("---------------------------------------------------------------------------------")
	# print("Best build model is: ")
	# print(max_build)
	# print("Build model score: " + str(max_score))
	# print("MAE = {:5.4f}".format(metrics.mean_absolute_error(y_test, predicted)))
	# print("MSE = {:5.4f}".format(metrics.mean_squared_error(y_test, predicted)))
	# print("RMSE = {:5.4f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
	# print("Score:", max_build.score(X_test, y_test))
	# print("---------------------------------------------------------------------------------")

	return max_build

# PNL History
def hist_to_pnl(coin_hist, start, order_amount):
	coin_hist["buy_q"] = coin_hist["go_signal"] * coin_hist["conviction_go"] * order_amount
	coin_hist["sell_q"] = coin_hist["sell_signal"] * coin_hist["conviction_sell"] * order_amount
	coin_hist["q"] = start + (coin_hist["buy_q"].cumsum() - coin_hist["sell_q"].cumsum())
	coin_hist["sell_rev"] = coin_hist["sell_q"] * coin_hist["sell_signal"] * coin_hist["btc_price"]
	coin_hist["buy_cost"] = coin_hist["buy_q"] * coin_hist["go_signal"] * coin_hist["btc_price"]
	coin_hist["TRev"] = coin_hist["sell_rev"].cumsum()
	coin_hist["TCost"] = coin_hist["buy_cost"].cumsum()
	coin_hist["curr_val"] = coin_hist["q"] * coin_hist["btc_price"]
	coin_hist["cash_flow"] = coin_hist["TRev"] - coin_hist["TCost"]
	coin_hist["hold_val"] = start * coin_hist["btc_price"]
	coin_hist["pnl"] = (coin_hist["curr_val"] + coin_hist["cash_flow"]) - coin_hist["hold_val"]
	coin_hist["algo_rt"] = (coin_hist["curr_val"] + coin_hist["cash_flow"])/coin_hist["hold_val"]
	coin_hist["trade_ind"] = pd.to_numeric(coin_hist["go_signal"] | coin_hist["sell_signal"])
	return coin_hist

index = ['bitcoin']

for x in index:
	import time
	import datetime
	time.sleep(1)
	filename = 'data/' + x + '_btc.csv'
	coin_price_hist(x, 'usd', 85, 'hourly').to_csv(filename)
	coin_hist = get_btc_hist_tms(x)
	cmd_log = pd.DataFrame(columns = ['market_tms', 'btc_price', 'p_go', 'p_sell', 'go_signal', 'sell_signal', 'algo_rt'])

	coin_hist = hist_to_model_prep(coin_hist)

	coin_hist = coin_hist.sort_values(by='market_tms', ascending=True)

	window = 28*24
	i = window
	while i < len(coin_hist):
		x = coin_hist.iloc[i - window : i ,:]
		y = coin_hist.iloc[i-1:i ,:]
		o = model_build_and_run(x)

		coin_hist["p_go"].iloc[i] =  o['p_go'].values[0]
		coin_hist["p_sell"].iloc[i] =  o['p_sell'].values[0]
		coin_hist["o3_go"].iloc[i] =  o['o3_go'].values[0]
		coin_hist["o3_sell"].iloc[i] =  o['o3_sell'].values[0]
		coin_hist["conviction_go"].iloc[i] =  o['conviction_go'].values[0]
		coin_hist["conviction_sell"].iloc[i] =  o['conviction_sell'].values[0]
		coin_hist["go_signal"].iloc[i] =  o['go_signal'].values[0]
		coin_hist["sell_signal"].iloc[i] =  o['sell_signal'].values[0]
		i = i + 1
		pct = round((i / len(coin_hist))*100, 2)
		if i % 79 == 0: print("Simulation is " + str(pct) + '% complete')

	coin_hist = hist_to_pnl(coin_hist, 1.5, 0.01)
	coin_hist = coin_hist.iloc[::-1]
	coin_hist.to_csv(('perf/bitcoin_perf.csv'))
	cmd = coin_hist[['market_tms', 'btc_price', 'p_go', 'p_sell', 'go_signal', 'sell_signal', 'buy_q', 'sell_q', 'buy_cost', 'sell_rev', 'algo_rt']]
	std = np.std(coin_hist["btc_price"])
	n = coin_hist["trade_ind"].sum()
	cmd['ntrades'] = n
	feed = cmd.head(1).iloc[0]
	print("----------------------------------------------------------------------------------------------")
	print(index)
	print("----------------------------------------------------------------------------------------------")
	print(feed)
	print(str(n))
  # Uncomment below to push orders to Coinbase Account
	# push_order(feed)
	get_fills_hist('ETH-BTC')
	print("----------------------------------------------------------------------------------------------")
	print("")
