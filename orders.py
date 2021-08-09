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


def get_fills_hist(market):
	import cbpro
	from authentication import (api_secret, api_key, api_pass)
	import json
	import pandas as pd
	url = 'https://api-public.sandbox.pro.coinbase.com'
	client = cbpro.AuthenticatedClient(api_key, api_secret, api_pass, api_url=url)
	fills_gen = client.get_fills(product_id=market)
	df = list(fills_gen)

	x = json.dumps(df)
	y = pd.read_json(x)
	z = pd.DataFrame()
	z['created_at'] = pd.to_datetime(y.created_at)
	z['trade_id'] = y.trade_id
	z['product_id'] = y.product_id
	z['price'] = y.price
	z['size'] = y['size']
	z['order_id'] = y.order_id 
	z['created_at'] = y.created_at
	z['liquidity'] = y.liquidity 
	z['fee'] = y.fee 
	z['settled'] = y.settled 
	z['side'] = y.side
	z.to_csv(('fills/' + market + '_fills.csv'))