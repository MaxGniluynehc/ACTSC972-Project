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
fig_PATH = "/Users/y222chen/Documents/Max/Study/ACTSC972-Finance3/Project/presentation/figs/"

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

(p1.loc[datetime(2013, 12, 30)]/p1.loc[datetime(1963,1,2)])**(1/50) # raw return over the trading period

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

# def daterange(start_date, end_date):
#     for n in range(int((end_date - start_date).days)):
#         yield start_date + timedelta(n)
#

base_test_timerange = list(df.index[(df.index > datetime(1963,1,1)) & (df.index < datetime(2013,12,31))])



MA_buy_signal = []
TSMOM_buy_signal = []
holding_day = []
for i, d in enumerate(base_test_timerange[:-1]):
    MA_buy_signal.append( int(df["P Lo 20"][d] > df["P Lo 20"][d-timedelta(days=9): d].mean()) )
    TSMOM_buy_signal.append( int(df["P Lo 20"][d] > df["P Lo 20"][d-timedelta(days=11): d-timedelta(days=9)].mean()) )
    holding_day.append( (base_test_timerange[i+1] - base_test_timerange[i]).days )


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

new_trade_period =(datetime(2014,1,1), datetime(2021,12,31)) #  (datetime(1963,1,1), datetime(1983,12,31)) #
Vt_bh, r_bh = get_portfolio_value_process(strategy="buy-hold", trade_period=new_trade_period)
Vt_MA, r_MA = get_portfolio_value_process(strategy="MA", trade_period=new_trade_period)
Vt_MOM, r_MOM = get_portfolio_value_process(strategy="TSMOM", trade_period=new_trade_period)

plt.figure()
plt.plot(Vt_bh, label= "buy and hold")
plt.plot(Vt_MA, label= "MA")
plt.plot(Vt_MOM, label= "MOM")
plt.xlim([datetime(1963,1,1), datetime(2001,1,1)])
plt.legend()


r_MA_50years = []
starts = []
for y in np.arange(1963, 2014):
    for m in np.arange(1,13):
        trade_period = (datetime(y,m,1), datetime(y,m,1) + relativedelta(months=12))
        V, r = get_portfolio_value_process(portfolio = "Lo 20", strategy="MA", look_back=100, transaction_cost = 0.4/100,
                                           trade_period=trade_period)
        r_MA_50years.append(r)
        starts.append(datetime(y,m,1))
r_MA_50years = pd.Series(r_MA_50years, index=pd.DatetimeIndex(starts))

len(r_MA_50years)
np.mean(r_MA_50years)
np.std(r_MA_50years)
np.mean(r_MA_50years) - 1.28 * np.std(r_MA_50years) # 10% significant level


r_MOM_50years = []
starts = []
for y in np.arange(1963, 2014):
    for m in np.arange(1,13):
        trade_period = (datetime(y,m,1), datetime(y,m,1) + relativedelta(months=12))
        V, r = get_portfolio_value_process(portfolio = "Lo 20", strategy="TSMOM", look_back=100, transaction_cost = 0.4/100,
                                           trade_period=trade_period)
        r_MOM_50years.append(r)
        starts.append(datetime(y,m,1))
r_MOM_50years = pd.Series(r_MOM_50years, index=pd.DatetimeIndex(starts))
len(r_MOM_50years)
np.mean(r_MOM_50years)
np.std(r_MOM_50years)
np.mean(r_MOM_50years) - 1.28 * np.std(r_MOM_50years) # 10% significant level

## correlation between MA and MOM returns: highly correlated
np.corrcoef(r_MA_50years, r_MOM_50years)


## Sharpe ratio

(np.mean(r_MA_50years) - 0.05)/np.std(r_MA_50years)

(np.mean(r_MOM_50years) - 0.05)/np.std(r_MOM_50years)


## Jensen's alpha
_,_,Rt = get_portfolio_value_process(portfolio="Qnt 2", strategy="MA", look_back=50,
                                     trade_period=(datetime(1963,1,1), datetime(2013,12,31)),
                                     return_Rt=True)
portfolio_return = Rt


def get_jensen_alpha(portfolio_return, return_fitted=False):
    '''
    :return: return the Jensen's alpha in % by fitted Fama-French 4 factor model.
    '''
    df_ff4 = ff4.loc[portfolio_return.index, :]
    Y = (portfolio_return-1)*100 - df_ff4["RF"]
    df_ff4["Y"] = Y  # (***)
    lm_formula = "Y ~ Mkt_RF + SMB + HML"
    lm_fitted = ols(lm_formula, df_ff4).fit()

    # Or from line (***), run the following:
    # X = df_ff4[["Mkt-RF", "SMB", "HML"]]
    # X = sm.add_constant(X, prepend=True).rename(columns={"const": "alpha"})
    # lm = sm.OLS(Y,X)
    # lm_fitted = lm.fit()
    alpha = lm_fitted.params[0] * 100
    if return_fitted:
        return lm_fitted
    else:
        return alpha


lm_fitted = get_jensen_alpha(Rt, return_fitted=True)
lm_fitted.summary()
# lm_fitted.t_test([0,1,-1,0])


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


Rt_monthly = convert_to_monthly_return(daily_return=Rt)

Rm_monthly=ff4m["Mkt"] [(ff4m["Mkt"] .index >= datetime(1963,1,1)) & (ff4m["Mkt"] .index <= datetime(2013,12,31))]

# plt.figure()
# plt.plot(Rm_monthly)

I_bear = np.array([(ff4m.loc[t - relativedelta(months=24): t, "Mkt"]).mean() < 0 for t in Rm_monthly.index]).astype(int)
# I_bear = np.array([all(ff4m.loc[t - relativedelta(months=5): t, "Mkt"] < 0) for t in Rm_monthly.index]).astype(int)
len(I_bear)

I_upmonth = np.array(ff4m.loc[Rm_monthly.index, "Mkt"] > 0).astype(int)
len(I_upmonth)

vol_m = np.array([(ff4.loc[t- timedelta(days=50): t, "Mkt"]).std() for t in Rm_monthly.index])
len(vol_m)

df_crash_analysis= ff4m.loc[Rm_monthly.index,:]
df_crash_analysis["I_bear"] = I_bear
df_crash_analysis["I_upmonth"] = I_upmonth
df_crash_analysis["vol_m"] = vol_m
df_crash_analysis["Y"] = (Rt_monthly-1)*100 - df_crash_analysis["RF"]

mkt_timing_formula = "Y ~ I_bear + Mkt_RF + I_bear:Mkt_RF + I_bear:I_upmonth:Mkt_RF"
lm_mkt_timing = ols(formula=mkt_timing_formula, data=df_crash_analysis).fit()
lm_mkt_timing.summary()


mkt_stress_formula = "Y ~ I_bear + vol_m + I_bear:vol_m"
lm_mkt_stress = ols(formula=mkt_stress_formula, data=df_crash_analysis).fit()
lm_mkt_stress.summary()












