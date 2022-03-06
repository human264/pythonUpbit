import pandas as pd
import pyupbit
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")

data = pd.read_csv("bitCoinUpbit.csv", parse_dates=["Date"], index_col="Date")
data["returns"] = np.log(data.Close / data.Close.shift(1))
print(data)

def backtest(data, parameters, tc):
    # prepare features
    data = data[["Close", "Volume", "returns"]].copy()
    data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
    data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
    data.loc[data.vol_ch < -3, "vol_ch"] = np.nan

    # define trading positions
    return_thresh = np.percentile(data.returns.dropna(), parameters[0])
    cond1 = data.returns >= return_thresh
    volume_thresh = np.percentile(data.vol_ch.dropna(), [parameters[1], parameters[2]])
    cond2 = data.vol_ch.between(volume_thresh[0], volume_thresh[1])

    data["position"] = 1
    data.loc[cond1 & cond2, "position"] = 0

    # backtest
    data["strategy"] = data.position.shift(1) * data["returns"]
    data["trades"] = data.position.diff().fillna(0).abs()
    data.strategy = data.strategy + data.trades * tc
    data["creturns"] = data["returns"].cumsum().apply(np.exp)
    data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

    # return strategy multiple
    return data.cstrategy[-1]

print(backtest(data= data, parameters=(90,5,20),tc = -0.0005))

return_range = range(85,98,1)
vol_low_range = range(2,16,1)
vol_high_range = range(16,35,1)

print(list(return_range))
combinations = list(product(return_range, vol_low_range, vol_high_range))
print(combinations)
print(len(combinations))


results = []
for comb in combinations:
    results.append(backtest(data = data, parameters = comb, tc= -0.0005))

many_results= pd.DataFrame(data = combinations, columns=["returns", "vol_low", "vol_high"])
many_results["performance"] = results

print(many_results)
print(many_results.nlargest(20, "performance"))
print(many_results.nsmallest(20, "performance"))

many_results.groupby("returns").performance.mean().plot()
plt.show()
many_results.groupby("vol_low").performance.mean().plot()
plt.show()
many_results.groupby("vol_high").performance.mean().plot()
plt.show()

print(backtest(data = data, parameters = (92, 14, 34), tc= -0.0005))
print(backtest(data = data, parameters = (85, 3, 28), tc= -0.0005))