import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from data_clearning import value_weighted_quintile_portfolio_returns_daily as df
from data_clearning import value_weighted_quintile_portfolio_returns_monthly as dfm
from data_clearning import tbill3m_daily as tb
from data_clearning import fama_french_4factors_daily as ff4
from data_clearning import fama_french_4factors_monthly as ff4m


import pandas.tseries.offsets as pdoffsets
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
import calendar
from utils import moving_average
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
# import getFamaFrenchFactors as gff # deprecated, directly input Fama-French factors from the website


# fig_PATH = "/Users/maxchen/Documents/Study/STA/ACTSC972-Finance3/Project/presentation/figs/"
fig_PATH = "/Users/maxchen/Documents/Study/STA/ACTSC972-Finance3/Project/project/figs/"

# fig_PATH = "/Users/y222chen/Documents/Max/Study/ACTSC972-Finance3/Project/presentation/figs/"

df.head()
tb.head()

p1 = np.cumprod(1+df["Lo 20"]/100)
p2 = np.cumprod(1+df["Qnt 2"]/100)
p3 = np.cumprod(1+df["Qnt 3"]/100)
p4 = np.cumprod(1+df["Qnt 4"]/100)
p5 = np.cumprod(1+df["Hi 20"]/100)

# portfolio value processes
plt.figure()
plt.plot(df.index, p1, label = "Lo 20")
plt.plot(df.index, p2, label = "Qnt 2")
plt.plot(df.index, p3, label = "Qnt 3")
plt.plot(df.index, p4, label = "Qnt 4")
plt.plot(df.index, p5, label = "Hi 20")
plt.legend(loc="upper left")
plt.xlim([datetime(1963,1,1), datetime(2022, 5, 1)])
# plt.ylim([0,30000])
plt.savefig(fig_PATH + "quintile_time_series.png")

#
# 3-month Tbill rate
plt.figure()
plt.plot(tb.DTB3, label="US 3-month Tbill")
plt.xlim([datetime(1980,1,1), datetime(2023, 12, 30)])
plt.ylabel("interest rate (%)")
plt.legend()
plt.savefig(fig_PATH + "tbill.png")
#

base_test_timerange = list(df.index[(df.index > datetime(1963,1,1)) & (df.index < datetime(2013,12,31))])



def get_portfolio_value_process(portfolio = "Hi 20", strategy="MA", look_back=100, transaction_cost = 0.4/100,
                                trade_period : tuple = (datetime(1963,1,1), datetime(2013,12,31)),
                                return_Rt=False):

    base_test_timerange = list(df.index[(df.index > trade_period[0]) & (df.index < trade_period[1])])

    Rt = (1 + df[portfolio][base_test_timerange] / 100)
    Rt[Rt.index[0]] -= transaction_cost # first day buy in

    if strategy == "buy-hold":
        Vt = np.cumprod(Rt)
        r = (Vt[-1] / Vt[0]) ** (365 / (Vt.index[-1] - Vt.index[0]).days) - 1
        if return_Rt:
            return Vt, r, Rt
        else:
            return Vt, r

    else:
        St = np.cumprod(Rt)
        St_ma = pd.Series(moving_average(St.values, look_back), index=St.index)
        if strategy == "MA":
            buy_signal = pd.Series((St[base_test_timerange] > St_ma[base_test_timerange]), dtype=int)
            holding_day = np.diff(buy_signal.index).astype("timedelta64[D]")
            holding_day = np.append(holding_day, 1).astype(dtype=int) # hold 1 day on the last trading day

            # buy_signal = []
            # holding_day = []
            #
            # for i, d in enumerate(base_test_timerange[:-1]):
            #     buy_signal.append(int(St[d] > St[d - timedelta(days=look_back): d].mean()))
            #     holding_day.append((base_test_timerange[i + 1] - base_test_timerange[i]).days)

        elif strategy == "TSMOM":
            buy_signal = pd.Series([St[i] > St[i - timedelta(days=look_back+1): i - timedelta(days=look_back-1)].mean()
                       for i in base_test_timerange], dtype=int, index=base_test_timerange)
            holding_day = np.diff(buy_signal.index).astype("timedelta64[D]")
            holding_day = np.append(holding_day, 1).astype(dtype=int) # hold 1 day on the last trading day


            # buy_signal = []
            # holding_day = []
            # for i, d in enumerate(base_test_timerange[:-1]):
            #     buy_signal.append(
            #         int(St[d] > St[d - timedelta(days=look_back+1): d - timedelta(days=look_back-1)].mean()))
            #     holding_day.append((base_test_timerange[i + 1] - base_test_timerange[i]).days)

        Rt = Rt * buy_signal
        no_buy_times = np.array(base_test_timerange)[np.array(buy_signal) == 0]
        change_position_times = np.array(base_test_timerange)[np.where(np.roll(np.array(buy_signal), 1) != np.array(buy_signal))[0]]
        Rt.loc[no_buy_times] = ((1 + tb.loc[no_buy_times, "DTB3"] / 100) ** (1 / 365)) ** (np.array(holding_day)[np.array(buy_signal) == 0])

        # Rt[change_position_times] -= transaction_cost
        Rt[change_position_times] = Rt[change_position_times] * (1-transaction_cost)
        Vt = np.cumprod(Rt)

        r = (Vt[-1] / Vt[0]) ** (365 / (Vt.index[-1] - Vt.index[0]).days) - 1
        if return_Rt:
            return Vt, r, Rt
        else:
            return Vt, r



def get_realized_volatility(Rt, method="annual"):
    if method == "daily":
        return np.sqrt(np.mean((Rt - np.mean(Rt))**2))
    elif method == "monthly":
        return np.sqrt(np.mean((Rt - np.mean(Rt))**2) *21)
    elif method == "annual":
        return np.sqrt(np.mean((Rt - np.mean(Rt))**2) *252)
    else:
        ValueError("Wrong method!")



def get_jensen_alpha(portfolio_return, return_fitted=False):
    '''
    :return: return the Jensen's alpha in % by fitted Fama-French 4 factor model.
    '''
    df_ff4 = ff4.loc[portfolio_return.index, :]
    Y = (portfolio_return-1)*100 - df_ff4["RF"]
    df_ff4["Y"] = Y  # (***)
    lm_formula = "Y ~ Mkt_RF + SMB + HML"
    lm_fitted = ols(lm_formula, df_ff4).fit()

    alpha = lm_fitted.params[0] * 100
    if return_fitted:
        return lm_fitted
    else:
        return alpha


# Crash analysis
def convert_to_monthly_return(daily_return):
    Rt = daily_return.copy()
    Rt.index = pd.MultiIndex.from_arrays([Rt.index.year, Rt.index.month], names=["year", "month"])

    Rt_monthly = Rt.groupby(Rt.index).prod()
    Rt_monthly.index = pd.MultiIndex.from_tuples(Rt_monthly.index, names=["year", "month"])

    Rt_monthly = Rt_monthly.reset_index()
    Rt_monthly["day"] = 1
    Rt_monthly["Date"] = pd.to_datetime(Rt_monthly[["year", "month", "day"]])
    Rt_monthly = Rt_monthly.set_index(keys="Date").drop(columns=["year", "month", "day"]).squeeze()
    return Rt_monthly













