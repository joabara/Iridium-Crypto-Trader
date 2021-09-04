# Iridium Trading Algorithm

## Executive Summary

* Iridium trading algorithms acheive consistently strong results across diference currencies, around a ~10% incremental alpha over holdout.

* Appetite control algorithm significantly reduces risks from market volatility and cashflow issues.

* By grouping many bots together into a Hive network we can diversify and smooth performance over time.


![](Decisions.gif)

*Orange leaves represent buying decisions, blue represents selling decisions*



```python
network.simSummary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): $356507.88
    Cash earned (Total Sell Revenue): $317059.55
    Net Cashflow: -39448.32
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Starting Asset Value: $11153.53
    Ending Assets Value: $71563.07
    Net Asset Value: $60409.54
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: $20961.22
    Return %: 187.93%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: $10222.95
    Algo Return over Hold%: 91.66%
    --------------------------------------


## Data

### Currency Index & Price


```python
from coin_hist_pull import *

index = import_coin_list()
index.head()
```




<div>
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
  </tbody>
</table>
</div>




```python
prices = coin_price_hist('bitcoin', 'usd', 85, 'hourly')
prices.head()
```




<div>
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
      <td>2021-09-04 15:29:44.000</td>
      <td>49972.512112</td>
      <td>9.397981e+11</td>
      <td>9.397981e+11</td>
    </tr>
    <tr>
      <th>2042</th>
      <td>2021-09-04 15:03:20.614</td>
      <td>50025.343668</td>
      <td>9.407907e+11</td>
      <td>9.407907e+11</td>
    </tr>
    <tr>
      <th>2041</th>
      <td>2021-09-04 14:07:26.127</td>
      <td>50157.218372</td>
      <td>9.453514e+11</td>
      <td>9.453514e+11</td>
    </tr>
    <tr>
      <th>2040</th>
      <td>2021-09-04 13:07:46.341</td>
      <td>50246.891663</td>
      <td>9.446783e+11</td>
      <td>9.446783e+11</td>
    </tr>
    <tr>
      <th>2039</th>
      <td>2021-09-04 12:06:26.440</td>
      <td>49958.691352</td>
      <td>9.397770e+11</td>
      <td>9.397770e+11</td>
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
      <th>2021-06-04</th>
      <td>74</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-05</th>
      <td>67</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-06</th>
      <td>71</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-07</th>
      <td>72</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2021-06-08</th>
      <td>94</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Methodology

### Individual Bot Strategy

A trading bot is defined by the two currencies it's going to trade. A BTC/USD bot will be a bot that will buy BTC in USD based on the prices of BTC. This means the bot can also trade USD/BTC. Let's init a bot that starts with 10,000 USD worth of Bitcoin and trades ~1,500 USD at a time.


```python
from Trader import *
inv_amt = 10000
pct = 0.15
bot = Trader('Iridium_bitcoin_usd', 'bitcoin', 'usd', 'CoinBasePro')
unit_cost = bot.hist['price'].iloc[42*24]
order_amount = inv_amt/unit_cost
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


    
![png](plots/output_11_0.png)
    


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
bot.hist_to_pnl(order_amount, order_amount*pct)
decisions = bot.coin_hist
decisions = decisions.iloc[42*24:]
decisions.to_csv('decisions.csv')
bot.summary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): $320461.38
    Cash earned (Total Sell Revenue): $311615.04
    Net Cashflow: -8846.35
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Start Quantity: 0.309 bitcoin
    Starting Asset Value: $11831.83
    Ending Quantity: 0.515 bitcoin
    Ending Assets Value: $25749.85
    Net Asset Value: $13918.03
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: 5071.68 usd
    Return %: 41.86%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: 1482.68 usd
    Algo Return over Hold%: 9.61%
    --------------------------------------


What this readout shows is that the BTC/USD bot was able to accumulate around 25,000 USD worth of BTC by making some smart trades and only spending a net of 8,846.35 USD. The bot, on paper, was able to double its asset value by adding 5071.68 USD in value.

However, this would not be a true performance readout. It is important to note that during this time period, the price of BTC/USD did increase quite drastically. We need to able to compare the performance of the strategy vs the holdout (i.e if we didn't do anything).

In conclusion, the algorithm was able to create 1482.68 USD in incremental net value, an incremental improvement rate of about 9.61%.

#### Resilience to Decreasing Prices

We validate the algorithm's robustness by see how algorithm return rate is affected by a drop in price. In this example, we will look at a USDT/BTC trading bot. Since the price of BTC in USD has gone up significantly in the past few weeks, the price of USDT in BTC will have gone down.


```python
from Trader import *
inv_amt = 0.1
pct = 0.15
bot = Trader('Iridium_USDT_BTC', 'tether', 'btc', 'CoinBasePro')
unit_cost = bot.hist['price'].iloc[42*24]
order_amount = inv_amt/unit_cost
bot.learn_and_sim(42*24)
bot.hist_to_pnl(order_amount, order_amount*pct)
bot.summary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): 1.09 BTC
    Cash earned (Total Sell Revenue): $1.16 BTC
    Net Cashflow: 0.07
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Start Quantity: $ 3210.766 tether
    Starting Asset Value: 0.08 BTC
    Ending Quantity: $ 65.127 tether
    Ending Assets Value: $0.0 BTC
    Net Asset Value: $-0.08 BTC
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: -0.01 BTC
    Return %: -12.84%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: 0.01 BTC
    Algo Return over Hold%: 12.53%
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




    <matplotlib.legend.Legend at 0x7fb7d9e8fc70>




    
![png](plots/output_18_1.png)
    


The plot above shows that the price of USDT in BTC drops immediately as the bot starts trading. On the flipside, the bot starts selling USDT for BTC, increasing the BTC Cashflow. This is effectively the same thing as buying a lot of BTC when the price increases to USDT.

The performance below shows that the algorithm below suffers a loss of ~13%. However, when comparing the performance versus the holdout (if the bot had done nothing at all), the bot still performed 13% better.

This means that while the bot takes a loss, it is able to react quickly and make sure the losses are minimized. While the USDT price tanked, it sold a lot of BTC to reduce the losses.

#### Hedging Risk with 'Appetite Control'

In early simulations, the bot was exceptionally profitable. It was very good at predicting price rises and subsequently putting in buy orders. However, after a three-week bull run, the bot would be extremely over-leveraged since it had been frantically buying as much as possible. Eventually, a market crash occurred, and the bot's performance went from ~ +80% to -30% in a day.  

To illustrate, you have 100,000 USD and 1 Bitcoin. After 3 weeks of price continuing to go up, you now have $0 and 3 Bitcoin. On paper, that's pretty good - you might have doubled your total asset value. But next week, the price of Bitcoin goes down 50 percent. You still have 3 Bitcoin, but now you have less value overall from when you started.

The way we fix this aggression is to create an 'appetite curve' to make sure the initial balance of currencies remains roughly the same. The more the bot buys, the less likely it is to buy more, yet still equally likely to sell. The inverse is also true.

To illustrate:


```python
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
control = np.random.normal(0, 25, 10000)
control = abs(control)
sns.distplot(control).set(title ="Appetite Control", xlabel="Off-Balance by N Trades", ylabel = "Likelihood To Continue Order")
```




    [Text(0.5, 1.0, 'Appetite Control'),
     Text(0.5, 0, 'Off-Balance by N Trades'),
     Text(0, 0.5, 'Likelihood To Continue Order')]




    
![png](plots/output_21_1.png)
    


So when the bot continues to be aggressively buying or selling, the actual amount it is allowed to trade is restricted until the balance sheet is in equilibrium.

### Hive Network


#### Three's a Party
As we mentioned before, the bots usually have a pretty good performance, but in some cases they are quite bad and are designed to focus on one market at time. We could have multiple bots trading in different markets, but there is a lot more value in storing them in a network. This accomplishes two important goals:

* Many bots trading differnt markets allows us to diversify investments. If one bot fails spectacularly it won't affect the overall performance.

* By putting all the bots in a network, we open up future opportunities to simulate collaboration between bots and improve computing performance.

We're going to take a sample of the top 10 coins in terms of market cap and we're going to create a network of bots to simulate orders. The simulation process takes a long time, but in production it won't be a problem since we are running only one record (the current hour) through the model and processing any orders that need to go through.

We will give each bot around 1,000 USD and a standard trade size of 15%, or 150 USD.


```python
from Hive import *
index['err'] = index['id'].apply(lambda x: 1 if '-' in x else 0)
index = index[index['err']==0]
index.head()
```




<div>
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
network.loadBotNet(index.iloc[0:10])
network.setWindow(42)
```


```python
network.learn_performance(42, 1000, 0.15)
network.networkToPerf()
```

#### Results
Now that we have run the network simulation, we can aggregate the performance across all bots in the network and view their results.


```python
network.simSummary()
```

    CASHFLOW
    --------------------------------------
    Cash spent (Total Buy Cost): $356507.88
    Cash earned (Total Sell Revenue): $317059.55
    Net Cashflow: -39448.32
    --------------------------------------
    
    ASSET VALUES: 
    --------------------------------------
    Starting Asset Value: $11153.53
    Ending Assets Value: $71563.07
    Net Asset Value: $60409.54
    --------------------------------------
    
    TEST RETURNS
    --------------------------------------
    PNL: $20961.22
    Return %: 187.93%
    --------------------------------------
    
    TEST VS HOLDOUT
    --------------------------------------
    PNL over Hold: $10222.95
    Algo Return over Hold%: 91.66%
    --------------------------------------


## Internal Collaboration (Next Steps)

While the trading algorithm is extremely effective on its own, it would be interesting to see if there are opportunities for collaborations between bots. Using a tool like `networkx` could help create links between bots.

One potential implementation would be to use an order correlation matrix to create links between bots. It would be interesting to see if some bots have high correlations of orders with each other. If that would be the case, we could save a lot of completxity by having only a few bots make decisions and others following.
