import pandas as pd

class Trader(object):
	def __init__(self, name, coin, coin2, cbp_market):
		# super(ClassName, self).__init__()
		self.coin = coin # bitcoin
		self.coin2 = coin2 # usd
		self.market = cbp_market
		self.name = name

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
		from coin_hist_pull import coin_price_hist
		from g_trends import g_trend_pull
		import warnings
		warnings.filterwarnings('ignore')


		hist = coin_price_hist(self.coin, self.coin2, 85, 'hourly')
		self.hist = hist.iloc[::-1]

		coin_hist = self.hist
		coin_hist["prev"] = coin_hist["usd_price"].shift(periods=1)
		coin_hist["usd_price2"] = coin_hist["usd_price"].rolling(2, win_type='triang').mean()
		coin_hist["usd_price3"] = coin_hist["usd_price"].rolling(3, win_type='triang').mean()
		coin_hist["usd_price4"] = coin_hist["usd_price"].rolling(4, win_type='triang').mean()
		coin_hist["usd_price5"] = coin_hist["usd_price"].rolling(5, win_type='triang').mean()
		coin_hist["usd_price6"] = coin_hist["usd_price"].rolling(6, win_type='triang').mean()
		coin_hist["usd_price7"] = coin_hist["usd_price"].rolling(7, win_type='triang').mean()
		coin_hist["usd_price8"] = coin_hist["usd_price"].rolling(8, win_type='triang').mean()
		coin_hist["usd_price9"] = coin_hist["usd_price"].rolling(9, win_type='triang').mean()
		coin_hist["usd_price10"] = coin_hist["usd_price"].rolling(10, win_type='triang').mean()
		coin_hist["usd_price11"] = coin_hist["usd_price"].rolling(11, win_type='triang').mean()
		coin_hist["usd_price12"] = coin_hist["usd_price"].rolling(12, win_type='triang').mean()
		coin_hist["usd_price13"] = coin_hist["usd_price"].rolling(13, win_type='triang').mean()
		coin_hist["usd_price14"] = coin_hist["usd_price"].rolling(14, win_type='triang').mean()
		coin_hist["usd_price15"] = coin_hist["usd_price"].rolling(15, win_type='triang').mean()
		coin_hist["usd_price16"] = coin_hist["usd_price"].rolling(16, win_type='triang').mean()
		coin_hist["usd_price17"] = coin_hist["usd_price"].rolling(17, win_type='triang').mean()
		coin_hist["usd_price18"] = coin_hist["usd_price"].rolling(18, win_type='triang').mean()
		coin_hist["usd_price19"] = coin_hist["usd_price"].rolling(19, win_type='triang').mean()
		coin_hist["usd_price20"] = coin_hist["usd_price"].rolling(20, win_type='triang').mean()
		coin_hist["usd_price24"] = coin_hist["usd_price"].rolling(24, win_type='triang').mean()
		coin_hist["usd_price72"] = coin_hist["usd_price"].rolling(72, win_type='triang').mean()
		coin_hist["next"] = coin_hist["usd_price12"].shift(periods=-24)
		coin_hist["n_bp"] = (coin_hist["next"]/coin_hist["usd_price"] - 1)
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

		g_trends = pd.DataFrame(g_trend_pull([self.coin]))
		coin_hist['dt'] = pd.to_datetime(coin_hist['market_tms'])
		coin_hist['dt'] = coin_hist['dt'].apply(lambda x: x.strftime("%Y-%m-%d"))
		g_trends['dt'] = g_trends.index
		g_trends['dt'] = g_trends['dt'].apply(lambda x: x.strftime("%Y-%m-%d"))
		coin_hist = pd.merge(coin_hist, g_trends, how="left", on=["dt"])

		coin_hist = coin_hist.fillna(method='ffill')
		coin_hist = coin_hist.dropna()
		coin_hist = coin_hist.sort_values(by='market_tms', ascending=True)

		self.coin_hist = coin_hist

	def learn_and_sim(self, window):
		i = window
		coin_hist = self.coin_hist
		while i < len(coin_hist):
			x = coin_hist.iloc[i - window : i ,:]
			y = coin_hist.iloc[i-1:i ,:]
			o = self.model_build_and_run(x)
			output = o[2]
			go_build = o[0]
			sell_build = o[1]

			coin_hist["p_go"].iloc[i] =  output['p_go'].values[0]
			coin_hist["p_sell"].iloc[i] =  output['p_sell'].values[0]
			coin_hist["o3_go"].iloc[i] =  output['o3_go'].values[0]
			coin_hist["o3_sell"].iloc[i] =  output['o3_sell'].values[0]
			coin_hist["conviction_go"].iloc[i] =  output['conviction_go'].values[0]
			coin_hist["conviction_sell"].iloc[i] =  output['conviction_sell'].values[0]
			coin_hist["go_signal"].iloc[i] =  output['go_signal'].values[0]
			coin_hist["sell_signal"].iloc[i] =  output['sell_signal'].values[0]
			i = i + 1
			pct = round((i / len(self.coin_hist))*100, 2)
			if i % 79 == 0: print("Simulation is " + str(pct) + '% complete')

		self.coin_hist = coin_hist

	def learn(self, window):
		i = window
		coin_hist = self.coin_hist
		daysago = len(coin_hist)-window-1
		yesterday = len(coin_hist)-1
		today = len(coin_hist)-1

		x = coin_hist.iloc[daysago : yesterday ,:]
		y = coin_hist.iloc[today ,:]
		o = self.model_build_and_run(x)
		output = o[2]
		self.go_build = o[0]
		self.sell_build = o[1]

		d =  [output['p_go'].values[0], output['p_sell'].values[0], output['o3_go'].values[0], 
		output['o3_sell'].values[0], output['conviction_go'].values[0], output['conviction_sell'].values[0], 
		output['go_signal'].values[0], output['sell_signal'].values[0]]

		self.decision =  d


	def model_build_and_run(self, coin_hist):
		from sklearn.preprocessing import StandardScaler
		from sklearn.model_selection import train_test_split
		import pandas as pd
		import numpy as np
		# coin_hist = self.coin_hist
		ft_col = ['usd_price', 'usd_price2', 'usd_price3', 'usd_price4', 'usd_price5', 'usd_price6',
		'usd_price7', 'usd_price8', 'usd_price9', 'usd_price10', 'usd_price11', 'usd_price12', 'usd_price13',
		'usd_price14', 'usd_price15', 'usd_price16', 'usd_price17', 'usd_price18', 'usd_price19', 'usd_price20',
		'usd_price24', 'usd_price72', self.coin]

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
		go_build = self.autoMachineLearning(X_train, X_test, y_train, y_test)
		p_go = pd.DataFrame(go_build.predict_proba(x)).iloc[:,0]
		p_go = np.array(p_go)
		o3_go = pd.DataFrame(go_build.predict_proba(y)).iloc[:,0]
		o3_go = np.array(o3_go)

		# Sell
		red = red.drop(columns=['go'])
		red['sell'] = np.array(coin_hist['sell'])
		x = red[red.columns[red.columns != 'sell']]
		X_train, X_test, y_train, y_test = train_test_split(x, red['sell'], test_size=0.25, random_state=0)
		sell_build = self.autoMachineLearning(X_train, X_test, y_train, y_test)
		p_sell = pd.DataFrame(sell_build.predict_proba(x)).iloc[:,0]
		p_sell = np.array(p_sell)
		o3_sell = pd.DataFrame(sell_build.predict_proba(y)).iloc[:,0]
		o3_sell = np.array(o3_sell)

		pgo_mean = np.mean(p_go)
		pgo_sd = np.std(p_go)
		psell_mean = np.mean(p_sell)
		psell_sd = np.std(p_sell)

		output["p_go"] = pgo_mean
		output["p_sell"] = psell_mean
		output["o3_go"] = o3_go
		output["o3_sell"] = o3_sell
		output["conviction_go"] = abs(o3_go)
		output["conviction_sell"] = abs(o3_sell)
		output["go_signal"] =  int(o3_go - o3_sell > 0.10)
		output["sell_signal"] = int(o3_sell - o3_go > 0.10)

		return go_build, sell_build, output

	def autoMachineLearning(self, X_train, X_test, y_train, y_test):
		from sklearn.linear_model import LogisticRegression
		from sklearn.svm import SVC, LinearSVC
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.tree import DecisionTreeClassifier
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.naive_bayes import GaussianNB
		from sklearn.linear_model import Perceptron
		from sklearn.neural_network import MLPClassifier
		from sklearn.linear_model import SGDClassifier
		from sklearn.ensemble import GradientBoostingClassifier
		from sklearn.model_selection import train_test_split
		from sklearn import metrics
		import numpy as np

		models_to_run = [DecisionTreeClassifier] #, GradientBoostingClassifier, SGDClassifier] #, SVR, MLPRegressor, GaussianNB, SGDRegressor, LogisticRegression, LinearSVC, ]
		max_score = 0
		max_build = 0
		max_RMSE = 9999
		for algo in models_to_run:
			build = self.build_models_on_train(algo, X_train, y_train)
			pred = build.predict(X_test)
			RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
			# print("MAE = {:5.4f}".format(metrics.mean_absolute_error(y_test, pred)))
			# print("MSE = {:5.4f}".format(metrics.mean_squared_error(y_test, pred)))
			# print("RMSE = {:5.4f}".format(np.sqrt(metrics.mean_squared_error(y_test, pred))))
			# print("Score:", build.score(X_test, y_test))
			# print()
			if RMSE < max_RMSE:
			# if build.score(X_test, y_test) > max_score:
				max_score = build.score(X_test, y_test)
				max_build = build
				max_RMSE = RMSE

		predicted = max_build.predict(X_test)

		# print()
		# print("RESULTS: ")
		# print("---------------------------------------------------------------------------------")
		# print("Best build model is: ")
		# print(str(max_build) + "   RMSE: " + str(max_RMSE) + "   Score: " + str(max_score))
		# print("Build model score: " + str(max_score))
		# print("MAE = {:5.4f}".format(metrics.mean_absolute_error(y_test, predicted)))
		# print("MSE = {:5.4f}".format(metrics.mean_squared_error(y_test, predicted)))
		# print("RMSE = {:5.4f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
		# print("Score:", max_build.score(X_test, y_test))
		# print("---------------------------------------------------------------------------------")

		return max_build

	def build_models_on_train(self, model, X_train,  y_train):
	    classifier = model()
	    classifier.fit(X_train, y_train)
	    return classifier

	def hist_to_pnl(self, start, order_amount):
		import pandas as pd
		coin_hist = self.coin_hist
		coin_hist["buy_q"] = coin_hist["go_signal"] * coin_hist["conviction_go"] * order_amount
		coin_hist["sell_q"] = coin_hist["sell_signal"] * coin_hist["conviction_sell"] * order_amount
		coin_hist["q"] = start + (coin_hist["buy_q"].cumsum() - coin_hist["sell_q"].cumsum())
		coin_hist["sell_rev"] = coin_hist["sell_q"] * coin_hist["sell_signal"] * coin_hist["usd_price"]
		coin_hist["buy_cost"] = coin_hist["buy_q"] * coin_hist["go_signal"] * coin_hist["usd_price"]
		coin_hist["TRev"] = coin_hist["sell_rev"].cumsum()
		coin_hist["TCost"] = coin_hist["buy_cost"].cumsum()
		coin_hist["curr_val"] = coin_hist["q"] * coin_hist["usd_price"]
		coin_hist["cash_flow"] = coin_hist["TRev"] - coin_hist["TCost"]
		coin_hist["hold_val"] = start * coin_hist["usd_price"]
		coin_hist["pnl"] = (coin_hist["curr_val"] + coin_hist["cash_flow"]) - coin_hist["hold_val"]
		coin_hist["algo_rt"] = (coin_hist["curr_val"] + coin_hist["cash_flow"])/coin_hist["hold_val"]
		coin_hist["trade_ind"] = pd.to_numeric(coin_hist["go_signal"] | coin_hist["sell_signal"])
		self.coin_hist = coin_hist

	def summary(self):
		import numpy as np
		coin_hist = self.coin_hist.iloc[::-1]
		coin_hist.to_csv(('perf/bitcoin_perf.csv'))
		cmd = coin_hist[['market_tms', 'usd_price', 'p_go', 'p_sell', 'go_signal', 'sell_signal', 'buy_q', 'sell_q', 'buy_cost', 'sell_rev', 'algo_rt']]
		std = np.std(coin_hist["usd_price"])
		n = coin_hist["trade_ind"].sum()
		cmd['ntrades'] = n
		feed = cmd.head(1).iloc[0]
		print("----------------------------------------------------------------------------------------------")
		print(self.coin)
		print("----------------------------------------------------------------------------------------------")
		print(feed)
		print(str(n))
		# push_order(feed)
		# get_fills_hist('BTC-USD')
		print("----------------------------------------------------------------------------------------------")
		print("")

	def push_order(feed):
		buy_cost = round(float(feed[8]),4)
		sell_rev = round(float(feed[9]),4)

		if (buy_cost > 0) or (sell_rev > 0):
			import cbpro
			from authentication import (api_secret, api_key, api_pass)
			from datetime import datetime
			now = datetime.now()
			url = 'https://api-public.sandbox.pro.coinbase.com'
			client = cbpro.AuthenticatedClient(api_key, api_secret, api_pass, api_url=url)
			order_transcript = "No Action"

			if( buy_cost > 0):
				order_transcript = str( client.place_market_order(product_id='ETH-BTC', side= 'buy', funds = buy_cost))

			if (sell_rev > 0):
				order_transcript = str(client.place_market_order(product_id='ETH-BTC', side= 'sell', funds = sell_rev))

			log = open("eth-btc-log.txt", "a")
			n = log.write((str(now) + " | " + "Buy: " + str(buy_cost) + "Sell: " + str(sell_rev) + " " + order_transcript + "\n"))
			log.close()
