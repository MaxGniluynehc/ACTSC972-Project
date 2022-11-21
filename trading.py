import numpy as np
import pandas as pd
from data_clearning import value_weighted_quintile_portfolio_returns_daily as df
from data_clearning import value_weighted_quintile_portfolio_returns_monthly as dfm
from data_clearning import tbill3m_daily as tb

from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
import calendar
from utils import moving_average

from scipy.stats import norm

fig_PATH = "/Users/maxchen/Documents/Study/STA/ACTSC972-Finance3/Project/presentation/figs/"

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
plt.xlim([datetime(1963,1,1), datetime(2013, 12, 30)])
plt.ylim([0,30000])
plt.savefig(fig_PATH + "quntile_time_series.png")

(p1.loc[datetime(2013, 12, 30)]/p1.loc[datetime(1963,1,2)])**(1/50)

# 3-month Tbill rate
plt.figure()
plt.plot(tb.DTB3)
plt.xlim([datetime(1980,1,1), datetime(2023, 12, 30)])


t0 = datetime(2008,1,25)
plt.figure()
plt.plot(p1.loc[t0:t0+timedelta(days=10)], label = "Lo 20")
plt.plot(p1.rolling(window=10).mean().loc[t0:t0+timedelta(days=10)], label="10-day MA")
plt.xticks([t0, t0+timedelta(days=10/2), t0+timedelta(days=10)],
           [t0.date(), (t0+timedelta(days=10/2)).date(), (t0+timedelta(days=10)).date()])
plt.legend()
plt.savefig(fig_PATH+"trading_strategy_demo_MA_TSMOM_same.png")

# MA Strategy
# base test 1963-2013
# back test 2014-2022


df["P Lo 20"] = p1
df["P Qnt 2"] = p2

# df["P Lo 20"][datetime(1973,1,1):datetime(2013,1,1)]
#
# df["P Lo 20"][datetime(1973,1,1)]
#
# def daterange(start_date, end_date):
#     for n in range(int((end_date - start_date).days)):
#         yield start_date + timedelta(n)
#
# [df["P Lo 20"][d] > df["P Lo 20"][d - timedelta(days=100)] for d in daterange(datetime(1973,1,1),datetime(2013,1,1))]


# int(df["P Lo 20"][datetime(1973,1,2)] > df["P Lo 20"][datetime(1973,1,2)-timedelta(days=100): datetime(1973,1,1)].mean())

base_test_timerange = list(df.index[(df.index > datetime(1963,1,1)) & (df.index < datetime(2013,12,31))])

type((base_test_timerange[3] - base_test_timerange[2]).days)

MA_buy_signal = []
TSMOM_buy_signal = []
holding_day = []
for i, d in enumerate(base_test_timerange[:-1]):
    MA_buy_signal.append( int(df["P Lo 20"][d] > df["P Lo 20"][d-timedelta(days=9): d].mean()) )
    TSMOM_buy_signal.append( int(df["P Lo 20"][d] > df["P Lo 20"][d-timedelta(days=11): d-timedelta(days=9)].mean()) )
    holding_day.append( (base_test_timerange[i+1] - base_test_timerange[i]).days )

# for i, d in enumerate(base_test_timerange[:-1]):
#     MA_buy_signal.append( int(df["P Qnt 2"][d] > df["P Qnt 2"][d-timedelta(days=100): d].mean()) )
#     TSMOM_buy_signal.append( int(df["P Qnt 2"][d] > df["P Qnt 2"][d-timedelta(days=101): d-timedelta(days=99)].mean()) )
#     holding_day.append( (base_test_timerange[i+1] - base_test_timerange[i]).days )

# transaction_cost = 0.4/100
#
# p1_return_acct_holding_days = (1 + df["Lo 20"][base_test_timerange[:-1]]/100) # ** np.array(holding_day)
# p1_return_acct_holding_days[p1_return_acct_holding_days.index[0]] -= transaction_cost
# p1_values = np.cumprod(p1_return_acct_holding_days)
#
# buy_and_hold_return = (p1_values[-1]/p1_values[0])**(365/(base_test_timerange[-1] - base_test_timerange[0]).days)
#
# p1_return_MA = p1_return_acct_holding_days * MA_buy_signal
#
# # np.argwhere(np.array(MA_buy_signal) == 0)
#
# MA_no_buy_times = np.array(base_test_timerange[:-1])[np.array(MA_buy_signal) == 0]
#
# # (np.array(holding_day)[np.array(MA_buy_signal) == 0])
#
# p1_change_position_times_MA = np.array(base_test_timerange[:-1])[np.where(np.roll(np.array(MA_buy_signal), 1) != np.array(MA_buy_signal))[0]]
#
# p1_return_MA.loc[MA_no_buy_times] = ((1+tb.loc[MA_no_buy_times,"DTB3"]/100)**(1/365.25)) ** (np.array(holding_day)[np.array(MA_buy_signal) == 0])
#
# p1_return_MA.loc[p1_change_position_times_MA] = p1_return_MA.loc[p1_change_position_times_MA] - transaction_cost
# # p1_return_MA.loc[MA_no_buy_times] = 1
#
# p1_values_MA = np.cumprod(p1_return_MA)
#
# MA_return = (p1_values_MA[-1]/p1_values_MA[0])**(365.25/(base_test_timerange[-1] - base_test_timerange[0]).days)


def get_portfolio_value_process(portfolio = "Lo 20", strategy="MA", look_back=10, transaction_cost = 0.4/100,
                                trade_period : tuple = (datetime(1963,1,1), datetime(2013,12,31))):

    base_test_timerange = list(df.index[(df.index > trade_period[0]) & (df.index < trade_period[1])])

    Rt = (1 + df[portfolio][base_test_timerange] / 100)
    Rt[Rt.index[0]] -= transaction_cost # first day buy in

    if strategy == "buy-hold":
        Vt = np.cumprod(Rt)
        r = (Vt[-1] / Vt[0]) ** (365 / (Vt.index[-1] - Vt.index[0]).days) - 1
        return Vt, r

    else:
        St = np.cumprod(Rt)
        St_ma = pd.Series(moving_average(St.values, look_back), index=St.index)
        if strategy == "MA":
            buy_signal = pd.Series((St[base_test_timerange] > St_ma[base_test_timerange]), dtype=int)
            holding_day = np.diff(buy_signal.index).astype("timedelta64[D]")

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
        return Vt, r


Vt_bh, r_bh = get_portfolio_value_process(strategy="buy-hold")
Vt_MA, r_MA = get_portfolio_value_process(strategy="MA")
Vt_MOM, r_MOM = get_portfolio_value_process(strategy="TSMOM")

plt.figure()
plt.plot(Vt_bh, label= "buy and hold")
plt.plot(Vt_MA, label= "MA")
plt.plot(Vt_MOM, label= "MOM")
plt.legend()

r_MA_50years = []
for i in range(1963, 2013):
    trade_period = [datetime(i,1,1), datetime(i,12,31)]
    V, r = get_portfolio_value_process(portfolio = "Lo 20", strategy="MA", look_back=100, transaction_cost = 0.4/100,
                                       trade_period=trade_period)
    r_MA_50years.append(r)

np.mean(r_MA_50years)
np.std(r_MA_50years)
np.mean(r_MA_50years) - 1.28 * np.std(r_MA_50years)

r_MOM_50years = []
for i in range(1963, 2013):
    trade_period = [datetime(i,1,1), datetime(i,12,31)]
    V, r = get_portfolio_value_process(portfolio = "Lo 20", strategy="TSMOM", look_back=100, transaction_cost = 0.4/100,
                                       trade_period=trade_period)
    r_MOM_50years.append(r)

np.mean(r_MOM_50years)
np.std(r_MOM_50years)

## correlation between MA and MOM returns: highly correlated
np.corrcoef(r_MA_50years, r_MOM_50years)


## Sharpe ratio
(np.mean(r_MOM_50years) - 0.05)/np.std(r_MOM_50years)


## Jensen's alpha





