class Trader(object):
	"""
	This class represents the node-level structure in the bot network. Each bot is responsible for trading in a specific market.
	Each bot is held accountable for maintaining their own record history. Once they have made decsions, they will send their 
	decisions to the greater HiveNet, who will process their order.
	"""
	def __init__(self, name, coin, coin2, cbp_market, inv_amt, a, control):
		import pandas as pd
		self.coin = coin # bitcoin
		self.coin2 = coin2 # usd
		self.init_price = coin_price_hist(self.coin, self.coin2, 1, 'hourly')[['market_tms', 'price']].sort_values(by='market_tms', ascending=False).head(1)['price'].values[0]
		self.market = cbp_market
		self.name = name
		self.balance = 0
		self.inv_amt = inv_amt
		self.a = a

		# Import queue
		try:
			queue = pd.read_csv('queue/'+self.name+'_queue.csv')
			self.buy_queue = queue[queue['order']=='buy']['market_tms'].values
			self.sell_queue = queue[queue['order']=='sell']['market_tms'].values
		except FileNotFoundError:
			print('FnF: Empty Queue')
			self.sell_queue = []
			self.buy_queue = []
		except EOFError:
			print('EOF: Empty Queue')
			self.sell_queue = []
			self.buy_queue = []

		# Import previous decisions, init if no previous decisions
		try: self.decisions = pd.read_csv('orders/'+self.name+'_decisions.csv')
		except FileNotFoundError:
			print('FnF: No previous decisions')
			self.decisions = pd.DataFrame(columns=['market_tms', 'price', 'conviction_go', 'conviction_sell', 'go_signal', 'sell_signal', 'coin', 'coin2', 'order_amount'])
		except EOFError:
			print('EOF: Empty Decision')
			self.decisions = pd.DataFrame(columns=['market_tms', 'price', 'conviction_go', 'conviction_sell', 'go_signal', 'sell_signal', 'coin', 'coin2', 'order_amount'])
		except Exception:
			self.decisions = pd.DataFrame(columns=['market_tms', 'price', 'conviction_go', 'conviction_sell', 'go_signal', 'sell_signal', 'coin', 'coin2', 'order_amount'])

		# Set control balance and orders dataframe
		self.control = control
		self.orders = pd.DataFrame(columns=['market_tms', 'price', 'conviction_go', 'conviction_sell', 'go_signal', 'sell_signal', 'coin', 'coin2', 'order_amount'])

		import warnings
		warnings.filterwarnings('ignore')

	def refresh_data(self):
		import pandas as pd
		import numpy as np
		import os
		import warnings
		warnings.filterwarnings('ignore')

		hist = coin_price_hist(self.coin, self.coin2, 85, 'hourly')
		self.hist = hist.iloc[::-1]

		# Create price momentum metrics
		coin_hist = self.hist
		coin_hist["prev"] = coin_hist["price"].shift(periods=1)
		coin_hist["price2"] = coin_hist["price"].rolling(2, win_type='triang').mean()
		coin_hist["price3"] = coin_hist["price"].rolling(3, win_type='triang').mean()
		coin_hist["price4"] = coin_hist["price"].rolling(4, win_type='triang').mean()
		coin_hist["price5"] = coin_hist["price"].rolling(5, win_type='triang').mean()
		coin_hist["price6"] = coin_hist["price"].rolling(6, win_type='triang').mean()
		coin_hist["price7"] = coin_hist["price"].rolling(7, win_type='triang').mean()
		coin_hist["price8"] = coin_hist["price"].rolling(8, win_type='triang').mean()
		coin_hist["price9"] = coin_hist["price"].rolling(9, win_type='triang').mean()
		coin_hist["price10"] = coin_hist["price"].rolling(10, win_type='triang').mean()
		coin_hist["price11"] = coin_hist["price"].rolling(11, win_type='triang').mean()
		coin_hist["price12"] = coin_hist["price"].rolling(12, win_type='triang').mean()
		coin_hist["price13"] = coin_hist["price"].rolling(13, win_type='triang').mean()
		coin_hist["price14"] = coin_hist["price"].rolling(14, win_type='triang').mean()
		coin_hist["price15"] = coin_hist["price"].rolling(15, win_type='triang').mean()
		coin_hist["price16"] = coin_hist["price"].rolling(16, win_type='triang').mean()
		coin_hist["price17"] = coin_hist["price"].rolling(17, win_type='triang').mean()
		coin_hist["price18"] = coin_hist["price"].rolling(18, win_type='triang').mean()
		coin_hist["price19"] = coin_hist["price"].rolling(19, win_type='triang').mean()
		coin_hist["price20"] = coin_hist["price"].rolling(20, win_type='triang').mean()
		coin_hist["price24"] = coin_hist["price"].rolling(24, win_type='triang').mean()
		coin_hist["price72"] = coin_hist["price"].rolling(72, win_type='triang').mean()

		# Create output variables (i.e will price go up)
		coin_hist["next"] = coin_hist["price12"].shift(periods=-24)
		coin_hist["n_bp"] = (coin_hist["next"]/coin_hist["price"] - 1)
		coin_hist["sell"] = coin_hist["n_bp"] > np.mean(coin_hist["n_bp"]) + 1*np.std(coin_hist["n_bp"])
		coin_hist["sell"] = coin_hist["sell"].astype('int32')
		coin_hist["go"] = coin_hist["n_bp"] < np.mean(coin_hist["n_bp"]) - 1*np.std(coin_hist["n_bp"])
		coin_hist["go"] = coin_hist["go"].astype('int32')

		# Init model outputs
		coin_hist["p_go"] = 0
		coin_hist["p_sell"] = 0
		coin_hist["o3_go"] =  0
		coin_hist["o3_sell"] =  0
		coin_hist["conviction_go"] = 0
		coin_hist["conviction_sell"] = 0
		coin_hist["sell_signal"] =  0
		coin_hist["go_signal"] = 0

		# Import google trends data from pytrends API
		g_trends = pd.DataFrame(g_trend_pull([self.coin]))
		coin_hist['dt'] = pd.to_datetime(coin_hist['market_tms'])
		coin_hist['dt'] = coin_hist['dt'].apply(lambda x: x.strftime("%Y-%m-%d"))
		g_trends['dt'] = g_trends.index
		g_trends['dt'] = g_trends['dt'].apply(lambda x: x.strftime("%Y-%m-%d"))
		coin_hist = pd.merge(coin_hist, g_trends, how="left", on=["dt"])

		# NA cleaning
		coin_hist = coin_hist.fillna(method='ffill')
		coin_hist = coin_hist.dropna()
		coin_hist = coin_hist.sort_values(by='market_tms', ascending=True)

		self.coin_hist = coin_hist

	def get_balance_control(self, balance):
		n = len(self.control)
		f = len(self.control[self.control > balance])
		return f/n

	def learn(self, window):
		import pandas as pd
		coin_hist = self.coin_hist
		i = len(coin_hist)-1
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
		now = coin_hist['market_tms'].iloc[i].replace(microsecond=0, second=0, minute=0)

		if now in self.sell_queue or i % window*24 == 0:
			coin_hist["conviction_go"].iloc[i] =  0 # output['conviction_go'].values[0]
			coin_hist["conviction_sell"].iloc[i] = (1-int(self.balance>0)*self.get_balance_control(self.balance)) 
			coin_hist["go_signal"].iloc[i] = 0 # output['go_signal'].values[0]
			coin_hist["sell_signal"].iloc[i] = 1 #  output['sell_signal'].values[0]
			if now in self.sell_queue: self.sell_queue.remove(now)
			self.balance-=1

		elif now in self.buy_queue:
			coin_hist["conviction_go"].iloc[i] = (1-int(self.balance<0)*self.get_balance_control(self.balance)) # output['conviction_go'].values[0]
			coin_hist["conviction_sell"].iloc[i] = 0 # int(self.balance>0)*self.get_self.balance_control(self.balance) # output['conviction_sell'].values[0]
			coin_hist["go_signal"].iloc[i] = 1 # output['go_signal'].values[0]
			coin_hist["sell_signal"].iloc[i] = 0 #  output['sell_signal'].values[0]
			self.buy_queue.remove(now)
			self.balance+=1

		else:
			coin_hist["conviction_go"].iloc[i] = output['conviction_go'].values[0]
			coin_hist["conviction_sell"].iloc[i] =output['conviction_sell'].values[0]
			coin_hist["go_signal"].iloc[i] = output['go_signal'].values[0]
			coin_hist["sell_signal"].iloc[i] = output['sell_signal'].values[0]
			future = (now+pd.Timedelta(days=window/2)).replace(microsecond=0, second=0, minute=0)

			if output['go_signal'].values[0] > 0:
				self.sell_queue.append(future)
				self.balance += 1
			if output['sell_signal'].values[0] > 0:
				self.buy_queue.append(future)
				self.balance -= 1

		self.coin_hist = coin_hist

		# Current decision
		self.decision = coin_hist.tail(1)[['market_tms', 'price', 'conviction_go', 'conviction_sell', 'go_signal', 'sell_signal']]
		self.decision['coin'] = self.coin
		self.decision['coin2'] = self.coin2
		self.decision['order_amount'] = (self.inv_amt/self.decision['price'].values[0])*self.a

		# Concat previous decisions
		self.decisions = pd.concat([self.decisions, self.decision], axis=0)
		self.decisions.to_csv('orders/'+self.name+'_decisions.csv')

		# Filter out pre-orders
		if self.decision['go_signal'].values[0] == 1 or self.decision['sell_signal'].values[0] == 1:
			self.orders = pd.concat([self.orders, self.decision], axis=0)


		# Save orders to queue
		buy = pd.DataFrame(self.buy_queue, columns=['market_tms'])
		buy['order'] = 'buy'
		sell = pd.DataFrame(self.sell_queue, columns=['market_tms'])
		sell['order'] = 'sell'
		self.queue = pd.concat([buy, sell], axis=0)
		self.queue.to_csv('queue/'+self.name+'_queue.csv')


	def model_build_and_run(self, coin_hist):
		from sklearn.preprocessing import StandardScaler
		from sklearn.model_selection import train_test_split
		import pandas as pd
		import numpy as np

		# Feature columns
		ft_col = ['price', 'price2', 'price3', 'price4', 'price5', 'price6',
		'price7', 'price8', 'price9', 'price10', 'price11', 'price12', 'price13',
		'price14', 'price15', 'price16', 'price17', 'price18', 'price19', 'price20',
		'price24', 'price72', self.coin]

		# Set up train pop and test pop
		X = coin_hist[ft_col]
		output = coin_hist[ft_col].tail(1)
		coin_hist.drop(coin_hist.tail(1).index, inplace=True)

		# Scale data to help fit model
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

		# Go - learning on buy indicators
		red['go'] = np.array(coin_hist['go'])
		x = red[red.columns[red.columns != 'go']]
		X_train, X_test, y_train, y_test = train_test_split(x, red['go'], test_size=0.25, random_state=0)
		go_build = self.autoMachineLearning(X_train, X_test, y_train, y_test)
		p_go = pd.DataFrame(go_build.predict_proba(x)).iloc[:,0]
		p_go = np.array(p_go)
		o3_go = pd.DataFrame(go_build.predict_proba(y)).iloc[:,0]
		o3_go = np.array(o3_go)

		# Sell - learning on sell indicators
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

		from sklearn.tree import DecisionTreeClassifier
		from sklearn.ensemble import RandomForestClassifier
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

	def orders_to_pnl(self):
		import pandas as pd
		start = self.inv_amt / self.init_price 
		order_amount = (self.inv_amt/start)*self.a
		perf = self.decisions
		perf["buy_q"] = perf["go_signal"] * perf["conviction_go"] * order_amount
		perf["sell_q"] = perf["sell_signal"] * perf["conviction_sell"] * order_amount
		perf["q"] = start + (perf["buy_q"].cumsum() - perf["sell_q"].cumsum())
		perf["sell_rev"] = perf["sell_q"] * perf["sell_signal"] * perf["price"]
		perf["buy_cost"] = perf["buy_q"] * perf["go_signal"] * perf["price"]
		perf["TRev"] = perf["sell_rev"].cumsum()
		perf["TCost"] = perf["buy_cost"].cumsum()
		perf["curr_val"] = perf["q"] * perf["price"]
		perf["cash_flow"] = perf["TRev"] - perf["TCost"]
		perf["hold_val"] = start * perf["price"]
		perf["pnl"] = (perf["curr_val"] + perf["cash_flow"]) - perf["hold_val"]
		perf["algo_rt"] = (perf["curr_val"] + perf["cash_flow"])/perf["hold_val"]
		perf["trade_ind"] = pd.to_numeric(perf["go_signal"] | perf["sell_signal"])
		self.perf = perf

	def summary(self):
		q1 = self.perf['q'].iloc[0]
		p1 = self.perf['price'].iloc[0]
		start_asset_val = float(q1)*float(p1)
		total_revenue = self.perf['sell_rev'].sum()
		total_cost = self.perf['buy_cost'].sum()
		q2 = self.perf['q'].tail(1).values[0]
		p2 = self.perf['price'].tail(1).values[0]
		total_asset_val = float(q2)*float(p2)
		pnl_over_hold = self.perf['pnl'].tail(1).values[0]
		pnl = total_asset_val-start_asset_val + total_revenue-total_cost
		a = 100*(((pnl+start_asset_val)/start_asset_val))-1
		a2 = 100*((pnl_over_hold+start_asset_val)/start_asset_val-1)
		print('CASHFLOW')
		print('--------------------------------------')
		print('Cash spent (Total Buy Cost): $' + str(round(total_cost,2)))
		print('Cash earned (Total Sell Revenue): $'+ str(round(total_revenue,2)))
		print('Net Cashflow: ' + str(round(total_revenue-total_cost, 2)))
		print('--------------------------------------')
		print()
		print('ASSET VALUES: ')
		print('--------------------------------------')
		print('Start Quantity: ' + str(round(q1,3)) + self.coin)
		print('Starting Asset Value: $' + str(round(start_asset_val, 2)))
		print('Ending Quantity: ' + str(round(q2,3)) + self.coin)
		print('Ending Assets Value: $' + str(round(total_asset_val,2)))
		print('Net Asset Value: $' + str(round(total_asset_val-start_asset_val,2)))
		print('--------------------------------------')
		print()
		print('TEST RETURNS')
		print('--------------------------------------')
		print('PNL: $' + str(round(pnl,2)))
		print('Return %: ' + str(round(a,2)) + '%')
		print('--------------------------------------')
		print()
		print('TEST VS HOLDOUT')
		print('--------------------------------------')
		print('PNL over Hold: $' + str(round(pnl_over_hold,2)))
		print('Algo Return over Hold%: ' +str(round(a2,2)) + '%')
		print('--------------------------------------')
