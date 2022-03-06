import pandas as pd
import pyupbit

import numpy as np
import matplotlib.pyplot as plt

# pyupbit.get_ohlcv("KRW-BTC","minutes60", 40000).to_csv("bitCoinUpbit.csv")


pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")

data = pd.read_csv("bitCoinUpbit.csv", parse_dates=["Date"], index_col="Date")
data = data[["Close", "Volume"]].copy()
# data.close.plot(figsize = (12, 8), title="BitCoin", fontsize = 12)
# plt.show()
# data.close.loc["2021-05"].plot(figsize = (12,8), title="BitCoin", fontsize = 12)
# plt.show()

data["returns"] = np.log(data.Close.div(data.Close.shift(1)))


print(data.describe())

data.returns.plot(kind="hist", bins = 100, figsize = (12,8))
plt.show()

print(data.returns.nlargest(10))
print(data.returns.nsmallest(10))

print(data.Close / data.Close[0])

print(data.returns.sum())

multiple = np.exp(data.returns.sum())
print(multiple)

data["creturns"] = data.returns.cumsum().apply(np.exp)
print(data)
data.Close.plot(figsize=(12,8), title = "BitCoin", fontsize = 12)
plt.show()

mu = data.returns.mean()
print(f"mu : {mu}")

std = data.returns.std()
print(f"std : {std}")

number_of_periods = 24*365.25

annual_mean = mu * number_of_periods
annual_std = std * np.sqrt(number_of_periods)

cagr = np.exp(annual_mean) -1

print(f"cagr : {cagr}")

#sharpe Ratio
print(annual_mean / annual_std)
print(cagr / annual_std)

data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
data.loc[data.vol_ch < -3, "vol_ch"] = np.nan
print(data)

data.vol_ch.plot(kind="hist", bins=100, figsize = (12,8))
plt.show()

print(data.vol_ch.nsmallest(20))
print(data.vol_ch.nlargest(20))

data.info()

print(data.info())

#Explanatory data analysis : finacial return and traidng vol
plt.scatter(x = data.vol_ch, y= data.returns)
plt.xlabel("Volume_change")
plt.ylabel("Returns")
plt.show()

pd.qcut(data.returns,  q= 10)

data["ret_cat"] = pd.qcut(data.returns, q = 10, labels = [-5,-4,-3,-2,-1,1,2,3,4,5])
data["vol_cat"] = pd.qcut(data.vol_ch, q = 10, labels = [-5,-4,-3,-2,-1,1,2,3,4,5])
print(data)

print(data.ret_cat.value_counts())

matrix = pd.crosstab(data.vol_cat, data.ret_cat)
print(matrix)

import seaborn as sns

plt.figure(figsize=(12,8))
sns.set(font_scale=1)
sns.heatmap(matrix, cmap = "RdYlBu_r", annot = True, robust = True, fmt = ".0f")
plt.show()

data.vol_cat.shift()

matrix = pd.crosstab(data.vol_cat.shift(), data.ret_cat.shift(), values=data.returns, aggfunc =np.mean)
print(matrix)
plt.figure(figsize=(15,12))
sns.set(font_scale=1)
sns.heatmap(matrix, cmap = "RdYlBu", annot = True, robust = True, fmt = ".5f")
plt.show()

data["position"] = 1

returns_thresh = np.percentile(data.returns.dropna(), 90)
print(returns_thresh)

cond1 = data.returns >= returns_thresh
print(cond1)

volume_thresh = np.percentile(data.vol_ch.dropna(), [5, 20])

print(volume_thresh)
cond2 = data.vol_ch.between(volume_thresh[0], volume_thresh[1])
print(cond2)

data.loc[cond1 & cond2, "position"] = 0
print(data)
print(data.position.value_counts())
data.loc[:, "position"].plot(figsize = (12,8))
plt.show()

data.loc["2022-02", "position"].plot(figsize = (12,12))
plt.show()

data["strategy"] = data.position.shift(1) * data["returns"]
print(data)

data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
print(data)

data[["creturns","cstrategy"]].plot(figsize=(12,8), fontsize= 12)
plt.show()
tp_year = 24*365.25

ann_mean = data[["returns", "strategy"]].mean() * tp_year
ann_std = data[["returns", "strategy"]].std() * np.sqrt(tp_year)

sharpe = (np.exp(ann_mean) - 1) / ann_std
print(f"ann_mean:{ann_mean}")
print(f"ann_std:{ann_std}")
print(f"sharpe :{sharpe}")

data.position.diff().fillna(0).abs()

data["trades"] = data.position.diff().fillna(0).abs()


print(data.trades.value_counts())

print(data)

commissions = 0.0005

ptc = np.log(1-commissions)

print(ptc)

data["strategy_net"] = data.strategy + data.trades * ptc
data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)

print(data)

data[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12,8))
plt.show()


data[["creturns", "cstrategy_net"]].plot(figsize = (12,8))
plt.show()

