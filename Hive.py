from Trader import *

class HiveNet(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.hive = []
		self.perf_grid = None
		self.decisions = []

	def setWindow(self, day_window):
		self.window = day_window

	def loadBotNet(self, index, inv_amt, a):
		import time
		self.index = []
		ref = dict({})
		for x in index['id']:
			try: 
				new_bot = Trader('Iridium-'+str(x)+'-USD', x, 'usd', 'Sim', inv_amt, a)
				self.hive.append(new_bot)
				time.sleep(2)
				self.index.append(new_bot.coin)
			except KeyError: pass # print(x + " doesn't work...")
			except ValueError: pass # print(x + " doesn't work...")

		for x in index.values:
			ref[x[0]] = x[1]
		self.ref = ref

	def learn_performance(self, day_window, inv_amt, pct):
		for bot in self.hive:
			unit_cost = bot.hist['price'].iloc[day_window*24]
			order_amount = inv_amt/unit_cost
			bot.learn_and_sim(day_window*24)


	def networkToPerf(self):
		total_revenue = 0
		total_cost = 0
		total_asset_val = 0
		start_asset_val = 0
		pnl = 0
		pnl2 = 0

		for bot in self.hive:
			q1 = bot.coin_hist['q'].iloc[self.window*24]
			p1 = bot.coin_hist['price'].iloc[self.window*24]
			start_asset_val += float(q1)*float(p1)
			total_revenue += bot.coin_hist['sell_rev'].sum()
			total_cost += bot.coin_hist['buy_cost'].sum()

			q2 = bot.coin_hist['q'].tail(1).values[0]
			p2 = bot.coin_hist['price'].tail(1).values[0]
			total_asset_val += float(q2)*float(p2)

			pnl2 += bot.coin_hist['pnl'].tail(1).values[0]
			pnl += float(q2)*float(p2)-float(q1)*float(p1) + (bot.coin_hist['sell_rev'].sum()-bot.coin_hist['buy_cost'].sum())

		self.total_revenue = total_revenue
		self.total_cost = total_cost
		self.total_asset_val = total_asset_val
		self.start_asset_val = start_asset_val
		self.pnl = pnl
		self.pnl2 = pnl2

	def learn(self, day_window):
		import pandas as pd
		self.decisions = pd.DataFrame(columns = ['coin','price','p_go', 'p_sell', 'o3_go', 'o3_sell', 'conviction_go', 'conviction_sell', 'go_signal', 'sell_signal', 'buy_q', 'sell_q'])
		for bot in self.hive:
			print(bot.coin + "-" + bot.coin2 + " processed...")
			self.decisions = pd.concat([self.decisions,bot.learn(day_window*24)])


	def process_orders(self):
		for bot in self.hive:
			buy = int(bot.decision[6]) == 1
			sell = int(bot.decision[6] == 1)
			if buy or sell :
				base_ticker = self.ref[bot.coin]
				tgt_ticker = self.ref[bot.coin2]
				print(base_ticker + "-" + tgt_ticker + ": " + str(bot.decision))

	def simSummary(self):
		a = 100*(self.pnl/self.start_asset_val)
		a2 = 100*(self.pnl2/self.start_asset_val)
		print('CASHFLOW')
		print('--------------------------------------')
		print('Cash spent (Total Buy Cost): $' + str(round(self.total_cost,2)))
		print('Cash earned (Total Sell Revenue): $'+ str(round(self.total_revenue,2)))
		print('Net Cashflow: ' + str(round(self.total_revenue-self.total_cost, 2)))
		print('--------------------------------------')
		print()
		print('ASSET VALUES: ')
		print('--------------------------------------')
		print('Starting Asset Value: $' + str(round(self.start_asset_val, 2)))
		print('Ending Assets Value: $' + str(round(self.total_asset_val,2)))
		print('Net Asset Value: $' + str(round(self.total_asset_val-self.start_asset_val,2)))
		print('--------------------------------------')
		print()
		print('TEST RETURNS')
		print('--------------------------------------')
		print('PNL: $' + str(round(self.pnl,2)))
		print('Return %: ' + str(round(a,2)) + '%')
		print('--------------------------------------')
		print()
		print('TEST VS HOLDOUT')
		print('--------------------------------------')
		print('PNL over Hold: $' + str(round(self.pnl2,2)))
		print('Algo Return over Hold%: ' +str(round(a2,2)) + '%')
		print('--------------------------------------')


# from coin_hist_pull import *
# index = import_coin_list()
# # index = pd.read_csv('data/00_coinlist.csv')['coin_id'][0:100]
# # index = ['ethereum', 'dogecoin', 'binancecoin', 'cardano', 'cosmos', 'chainlink', 'tether']
# network = HiveNet()
# network.loadBotNet(index.iloc[0:30]) # .iloc[0:30]
# network.setWindow(42)
# network.learn_performance(42, 1000, 0.003)
# network.learn(42)
# network.networkToPerf()
# network.perf.to_csv('hive_perf.csv')