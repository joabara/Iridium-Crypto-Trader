def g_trend_pull(keywords):
	from pytrends.request import TrendReq
	pytrend = TrendReq()
	pytrend.build_payload(kw_list=keywords, timeframe='today 3-m')
	df = pytrend.interest_over_time()
	return df