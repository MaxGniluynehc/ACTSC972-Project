import numpy as np
import pandas as pd
from data_clearning import value_weighted_quintile_portfolio_returns as df

from matplotlib import pyplot as plt
from datetime import datetime

df.head()

p1 = np.cumprod(1+df["Lo 20"]/100)
p2 = np.cumprod(1+df["Qnt 2"]/100)
p3 = np.cumprod(1+df["Qnt 3"]/100)
p4 = np.cumprod(1+df["Qnt 4"]/100)
p5 = np.cumprod(1+df["Hi 20"]/100)


# portfolio value processes
plt.figure()
plt.plot(df.Date, p1, label = "Lo 20")
plt.plot(df.Date, p2, label = "Qnt 2")
plt.plot(df.Date, p3, label = "Qnt 3")
plt.plot(df.Date, p4, label = "Qnt 4")
plt.plot(df.Date, p5, label = "Hi 20")
plt.legend()
plt.xlim([datetime(1980,1,1), datetime(2023, 12, 30)])





















