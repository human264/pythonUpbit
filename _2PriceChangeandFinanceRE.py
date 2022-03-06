#Price change and Finalail returns
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")

close = pd.read_csv("close.csv", index_col="Date", parse_dates=["Date"])

print(close)

hm = close.HYUNDAI_MOBILE.dropna().to_frame().copy()

hm.rename(columns = {"HYUNDAI_MOBILE":"Price"}, inplace=True)
print(hm)

hm["OneDayShifted"] = hm.shift(periods = 1)
hm["differ"] = hm.Price.sub(hm.OneDayShifted)
hm["differ2"] = hm.Price.diff(periods = 1)

print(hm.Price.div(hm.OneDayShifted) - 1)

hm["Returns"] = hm.Price.pct_change(periods = 1)

hm.drop(columns= ["OneDayShifted", "differ", "differ2"], inplace=True)
print(hm)

hm.to_csv("hm.csv")