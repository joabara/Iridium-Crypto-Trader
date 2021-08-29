import pandas as pd

def coin_price_hist(coin_id, vs_currency, days, interval):
    import pandas as pd
    import json
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()

    df = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days, interval=interval)
    x = json.dumps(df)
    y = pd.read_json(x)
    z = pd.DataFrame()
    z['market_tms'] = pd.to_datetime(pd.to_numeric(y.prices.str[0]), unit='ms')
    z['price'] = y.prices.str[1]
    z['mkt_cap'] = y.market_caps.str[1]
    z['total_volumes'] = y.market_caps.str[1]
    z = z.sort_values(by=['market_tms'], ascending=False)
    return (z)


def coin_candle_hist(coin_id, vs_currency, days):
    import pandas as pd
    import json
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    df = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    x = json.dumps(df)
    y = pd.read_json(x)
    z = pd.DataFrame()
    z['market_tms'] = pd.to_datetime(y[0], unit='ms')
    z['open'] = y[1]
    z['high'] = y[2]
    z['low'] = y[3]
    z['close'] = y[4]
    z = z.sort_values(by=['market_tms'], ascending=False)
    return (z)


def get_binance_exchange_info():
    import pandas as pd
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()

    binance_exchange = pd.DataFrame(columns=['base', 'target', 'market', 'last',
                                             'volume', 'converted_last', 'converted_volume',
                                             'trust_score', 'bid_ask_spread_percentage', 'timestamp',
                                             'last_traded_at', 'last_fetch_at', 'is_anomaly', 'is_stale',
                                             'trade_url', 'token_info_url', 'coin_id', 'target_coin_id'])

    for i in range(1, 11):
        df = cg.get_exchanges_tickers_by_id(id='binance', page=i, depth=True)
        page = pd.DataFrame(df['tickers'],
                            columns=['base', 'target', 'market', 'last',
                                     'volume', 'converted_last', 'converted_volume',
                                     'trust_score', 'bid_ask_spread_percentage', 'timestamp',
                                     'last_traded_at', 'last_fetch_at', 'is_anomaly', 'is_stale',
                                     'trade_url', 'token_info_url', 'coin_id', 'target_coin_id'])
        frames = [binance_exchange, page]

        binance_exchange = pd.concat(frames)
    return(binance_exchange)


def import_coin_list():
    import pandas as pd
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    df = cg.get_coins_markets('usd')
    coin_list = pd.DataFrame(df, columns=['id', 'symbol', 'name'])
    coin_list['symbol'] = coin_list['symbol'].apply(lambda x: x.upper())
    return coin_list

def get_coin_hist_tms(coin):
	hist = pd.read_csv("data/" + coin +  "_btc.csv")
	hist = hist.iloc[::-1]
	return hist