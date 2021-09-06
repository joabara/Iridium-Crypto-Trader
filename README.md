# Iridium Trading Hive

## Executive Summary

* Iridium trading algorithms acheive consistently strong results across diference currencies, like a 20% incremental alpha over holdout.

* Appetite control algorithm significantly reduces risks from market volatility and cashflow issues.

* By grouping many bots together into a Hive network we can diversify and smooth performance over time.


```python
network.simSummary()
```

## Data

### Currency Index & Price


```python
from coin_hist_pull import *

index = import_coin_list()
index.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>symbol</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ethereum</td>
      <td>ETH</td>
      <td>Ethereum</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cardano</td>
      <td>ADA</td>
      <td>Cardano</td>
    </tr>
    <tr>
      <th>3</th>
      <td>binancecoin</td>
      <td>BNB</td>
      <td>Binance Coin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tether</td>
      <td>USDT</td>
      <td>Tether</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ripple</td>
      <td>XRP</td>
      <td>XRP</td>
    </tr>
    <tr>
      <th>6</th>
      <td>solana</td>
      <td>SOL</td>
      <td>Solana</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dogecoin</td>
      <td>DOGE</td>
      <td>Dogecoin</td>
    </tr>
    <tr>
      <th>8</th>
      <td>polkadot</td>
      <td>DOT</td>
      <td>Polkadot</td>
    </tr>
    <tr>
      <th>9</th>
      <td>usd-coin</td>
      <td>USDC</td>
      <td>USD Coin</td>
    </tr>
  </tbody>
</table>
</div>




```python
prices = coin_price_hist('bitcoin', 'usd', 85, 'hourly')
prices.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market_tms</th>
      <th>price</th>
      <th>mkt_cap</th>
      <th>total_volumes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2043</th>
      <td>2021-09-06 13:36:27.000</td>
      <td>51236.488363</td>
      <td>9.635965e+11</td>
      <td>9.635965e+11</td>
    </tr>
    <tr>
      <th>2042</th>
      <td>2021-09-06 13:03:42.420</td>
      <td>51328.326739</td>
      <td>9.653960e+11</td>
      <td>9.653960e+11</td>
    </tr>
    <tr>
      <th>2041</th>
      <td>2021-09-06 12:00:58.225</td>
      <td>51384.962032</td>
      <td>9.664615e+11</td>
      <td>9.664615e+11</td>
    </tr>
    <tr>
      <th>2040</th>
      <td>2021-09-06 11:06:58.999</td>
      <td>51770.081701</td>
      <td>9.730119e+11</td>
      <td>9.730119e+11</td>
    </tr>
    <tr>
      <th>2039</th>
      <td>2021-09-06 10:01:58.312</td>
      <td>51810.725514</td>
      <td>9.744632e+11</td>
      <td>9.744632e+11</td>
    </tr>
  </tbody>
</table>
</div>



### Google Trends


```python
from g_trends import *
btc_searches = g_trend_pull(['bitcoin'])
btc_searches.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bitcoin</th>
      <th>isPartial</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-06-06</th>
      <td>70</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-07</th>
      <td>67</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-08</th>
      <td>91</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-09</th>
      <td>86</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-10</th>
      <td>75</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Methodology

### Individual Bot Strategy

A trading bot is defined by the two currencies it's going to trade. A BTC/USD bot will be a bot that will buy BTC in USD based on the prices of BTC. This means the bot can also trade USD/BTC. Let's init a bot that starts with 1000 USD worth of Bitcoin and trades ~400 USD at a time.


```python
from Trader import *
inv_amt = 1000
pct = 0.40
bot = Trader('Iridium_bitcoin_usd', 'bitcoin', 'usd', 'CoinBasePro', inv_amt, pct)
bot.learn_and_sim(42*24)
```


```python
from sklearn.tree import plot_tree
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
figure(figsize=(14, 12), dpi=225)
w = bot.go_build
plot_tree(w, filled=True)
plt.show()
```


    
![png](output_11_0.png)
    


#### Trading Strategy

Every hour, the bot looks at the previous 42 days and identifies what the best opportunities to buy and sell would have been. This will set up the training data for the machine learning algorithm later.

Specifically, there are two labels that the bot learns for:

* Buy Ind: If the price change increases by more than 1 standard deviation over the next hour
* Sell Ind: If the price change decreases by more than 1 standard deviation over the next hour

The features can be simplified to:
* Prices and moving averages of prices (momentum)
* Google Trends data (how much public interest is in a currency)

The training data will be the last 42 days minus the current hour, and then split on a 75/25 size. The current hour's data will then be run through the model build and it will return a decision: buy, sell, or do nothing.

#### Results From Individual Trader

After a couple weeks of trading, the bot's performance in terms of PNL is usually profitable (depending on the currency). Most of the value comes from increased asset size while maintaining a small deficit in cashflow.

Using the example from before, if the bot starts with 100,000 USD and 1 Bitcoin, the ending balance sheet might look like 87,000 USD and 4 Bitcoin. So in essence, you would have bought 3 Bitcoin for 13,000 USD, which is a pretty great deal.

Below, is a the simulation result from a BTC/USD trading bot.


```python
bot.summary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): 36680.6  usd
    Cash earned (Total Sell Revenue): $38153.44  usd
    Net Cashflow: 1472.84  usd
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Start Quantity: 0.028 bitcoin
    Starting Asset Value: 1101.74  usd
    Ending Quantity: 0.006 bitcoin
    Ending Assets Value: 303.08  usd
    Net Asset Value: $-798.66
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: 674.18 usd
    Return %: 60.19%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: 352.82 usd
    Algo Return over Hold%: 24.79%
    --------------------------------------


What this readout shows is that the BTC/USD bot was able to accumulate around 17,000 USD worth of BTC by making some smart trades and only spending a net of 9,300 USD. The bot, on paper, was able to double its asset value by adding 8,500 USD in value.

However, this would not be a true performance readout. It is important to note that during this time period, the price of BTC/USD did increase quite drastically. We need to able to compare the performance of the strategy vs the holdout (i.e if we didn't do anything).

In conclusion, the algorithm was able to create ~ 2,000 USD in net value, an incremental improvement rate of about 20%.

#### Resilience to Decreasing Prices

We validate the algorithm's robustness by see how algorithm return rate is affected by a drop in price. In this example, we will look at a USDT/ETH trading bot. Since the price of ETH in USD has gone up significantly in the past few weeks, the price of USDT in ETH will have gone down.


```python
from Trader import *
inv_amt = 0.05
pct = 0.40
bot = Trader('Iridium_bitcoin_usd', 'tether', 'btc', 'CoinBasePro', inv_amt, pct)
bot.learn_and_sim(42*24)
bot.coin_hist.iloc[42*24:].to_csv('decisions2.csv')
bot.summary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): 0.97  btc
    Cash earned (Total Sell Revenue): $0.93  btc
    Net Cashflow: -0.04  btc
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Start Quantity: 1795.742 tether
    Starting Asset Value: 0.05  btc
    Ending Quantity: 4449.449 tether
    Ending Assets Value: 0.09  btc
    Net Asset Value: $0.04
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: -0.0 btc
    Return %: -6.81%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: 0.01 btc
    Algo Return over Hold%: 21.99%
    --------------------------------------



```python
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
%matplotlib inline

ts = bot.coin_hist[['price', 'dt', 'curr_val', 'cash_flow']].iloc[42*24:]
ts = ts.groupby(['dt']).mean()
ts.index = pd.to_datetime(ts.index)
fig, ax = plt.subplots(figsize=(10, 6))
axb = ax.twinx()

ax.plot(ts.index.values, ts.price, color='tab:blue', linestyle='-', label='Price')
axb.plot(ts.index.values, ts.cash_flow, color='tab:green', linestyle='-', label='Cash Flow')

axb.set_ylabel('BTC Cashflow')

ax.set_xlabel('Date')
ax.set_ylabel('Price of $1 in BTC')
ax.set_title('USDT / BTC')

# format x axis labels
plt.setp(ax.get_xticklabels(), rotation=90)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(DayLocator())

ax.legend(loc='upper left')
axb.legend(loc='upper right')
```




    <matplotlib.legend.Legend at 0x7f890ce42280>




    
![png](output_18_1.png)
    


The plot above shows that the price of USDT in ETH drops immediately as the bot starts trading. On the flipside, the bot starts selling USDT for ETH, increasing the ETH Cashflow. This is effectively the same thing as buying a lot of ETH when the price increases to USDT.

The performance below shows that the algorithm below suffers a loss of ~11%. However, when comparing the performance versus the holdout (if the bot had done nothing at all), the bot still performed 55% better.

This means that while the bot takes a loss, it is able to react quickly and make sure the losses are minimized. While the USDT price tanked, it sold a lot of ETH to reduce the losses.

### Hive Network


#### Three's a Party
As we mentioned before, the bots usually have a pretty good performance, but in some cases they are quite bad and are designed to focus on one market at time. We could have multiple bots trading in different markets, but there is a lot more value in storing them in a network. This accomplishes two important goals:

* Many bots trading differnt markets allows us to diversify investments. If one bot fails spectacularly it won't affect the overall performance.

* By putting all the bots in a network, we open up future opportunities to simulate collaboration between bots and improve computing performance.

We're going to take a sample of the top 10 coins in terms of market cap and we're going to create a network of bots to simulate orders. The simulation process takes a long time, but in production it won't be a problem since we are running only one record (the current hour) through the model and processing any orders that need to go through.

We will give each bot around 1,000 USD and a standard trade size of 2.5%, or 25 USD.


```python
from Hive import *
index['err'] = index['id'].apply(lambda x: 1 if '-' in x else 0)
index = index[index['err']==0]
index.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>symbol</th>
      <th>name</th>
      <th>err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ethereum</td>
      <td>ETH</td>
      <td>Ethereum</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cardano</td>
      <td>ADA</td>
      <td>Cardano</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>binancecoin</td>
      <td>BNB</td>
      <td>Binance Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tether</td>
      <td>USDT</td>
      <td>Tether</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
network = HiveNet()
network.loadBotNet(index.iloc[0:5], 500, 0.40)
network.setWindow(42)
```


```python
network.learn_performance(42, 500, 0.4)
network.networkToPerf()
```

#### Results
Now that we have run the network simulation, we can aggregate the performance across all bots in the network and view their results.


```python
network.simSummary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): $90002.06
    Cash earned (Total Sell Revenue): $90694.37
    Net Cashflow: 692.32
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Starting Asset Value: $2449.01
    Ending Assets Value: $3424.97
    Net Asset Value: $975.97
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: $1668.29
    Return %: 68.12%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: $343.31
    Algo Return over Hold%: 14.02%
    --------------------------------------


## Internal Collaboration (Next Steps)

While the trading algorithm is extremely effective on its own, it would be interesting to see if there are opportunities for collaborations between bots. Using a tool like `networkx` could help create links between bots.

One potential implementation would be to use an order correlation matrix to create links between bots. It would be interesting to see if some bots have high correlations of orders with each other. If that would be the case, we could save a lot of completxity by having only a few bots make decisions and others following.
