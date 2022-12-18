
import numpy as np
import pandas as pd
import itertools
import scipy.stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.formula.api import ols

from data_clearning import value_weighted_quintile_portfolio_returns_daily as df
from data_clearning import value_weighted_quintile_portfolio_returns_monthly as dfm
from data_clearning import tbill3m_daily as tb
from data_clearning import fama_french_4factors_daily as ff4
from data_clearning import fama_french_4factors_monthly as ff4m

from trading import get_portfolio_value_process,  get_realized_volatility, \
    get_jensen_alpha, convert_to_monthly_return

import pandas.tseries.offsets as pdoffsets
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
import calendar
from utils import moving_average
from dateutil.relativedelta import relativedelta


fig_PATH = "/Users/maxchen/Documents/Study/STA/ACTSC972-Finance3/Project/project/figs/"
quintile_list = ["Lo 20", "Qnt 2", "Qnt 3", "Qnt 4", "Hi 20"]
trading_strategies = ["buy-hold", "MA", "TSMOM"]
lookback_list = [20, 50, 100, 200]
df.head()

transaction_cost = 0.4/100 # 0.3/100
trade_period = (datetime(1963, 1, 1), datetime(1983, 12, 31))  # (datetime(2005,1,1), datetime(2021,12,31)) #


# ================================= Plot value/return processes ================================= #
# 3 trading strategies: buy-and-hold, MA, TSMOM
# 5 quintile portfolios: Lo 20, Qnt 2, Qnt 3, Qnt 4, Hi 20
# 4 lookback periods: 20, 50, 100, 200

for qnt in quintile_list:
    for lb in lookback_list:
        Vt_bh, r_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb)
        Vt_MA, r_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb)
        Vt_MOM, r_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=trade_period,
                                                    transaction_cost=transaction_cost, look_back=lb)

        plt.figure(figsize=[5,4])
        plt.plot(Vt_bh, label= "buy and hold, r={}%".format(round(r_bh*100,2)))
        plt.plot(Vt_MA, label= "MA, r={}%".format(round(r_MA*100,2)))
        plt.plot(Vt_MOM, label= "MOM, r={}%".format(round(r_MOM*100,2)))
        plt.legend()
        # plt.savefig(fig_PATH+"selfrep_Hi20_Vt_20052021.png")
        # plt.savefig(fig_PATH+"Vt_19631983_lb={}_pf={}.png".format(lb, qnt))
        plt.savefig(fig_PATH+"Vt_20052021_lb={}_pf={}.png".format(lb, qnt))



        fig, ax = plt.subplots(3,1, sharex=True, figsize=[5,4])
        ax[0].plot(Vt_bh.index[:-1], (Vt_bh.values[1:]- Vt_bh.values[:-1])/Vt_bh.values[:-1], color="blue", label= "buy and hold")
        ax[0].legend()
        ax[1].plot(Vt_bh.index[:-1], (Vt_MA.values[1:]- Vt_MA.values[:-1])/Vt_MA.values[:-1], color="orange", label= "MA")
        ax[1].legend()
        ax[2].plot(Vt_bh.index[:-1], (Vt_MOM.values[1:]- Vt_MOM.values[:-1])/Vt_MOM.values[:-1], color="green", label= "MOM")
        ax[2].legend()
        # plt.xlim([datetime(1963,1,1), datetime(2001,1,1)])
        # plt.savefig(fig_PATH+"selfrep_Lo20_Rt_20052021.png")
        # plt.savefig(fig_PATH + "Rt_19631983_lb={}_pf={}.png".format(lb, qnt))
        plt.savefig(fig_PATH + "Rt_20052021_lb={}_pf={}.png".format(lb, qnt))


# ================================= Return Correlations ================================= #
qnt = next(iter(quintile_list))
lb = next(iter(lookback_list))
corr_df = pd.DataFrame(columns=["buy-hold vs MA", "buy-hold vs MA", "MA vs TSMOM"],
                       index=list(itertools.product(quintile_list, lookback_list)))

i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=trade_period,
                                                    transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

        Rt = np.column_stack((Rt_bh.values, Rt_MA.values, Rt_MOM.values))
        corr_mat = np.corrcoef(Rt.T)
        corr_df.iloc[i,:] = [corr_mat[0,1], corr_mat[0,2], corr_mat[1,2]]
        i += 1
print(round(corr_df,1).to_latex())


# ================================= annualized Returns ================================= #
qnt = next(iter(quintile_list))
lb = next(iter(lookback_list))
rt_df = pd.DataFrame(columns=["buy-hold", "MA", "TSMOM"],
                       index=list(itertools.product(quintile_list, lookback_list)))

i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=trade_period,
                                                    transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

        RV_bh = get_realized_volatility(Rt_bh)
        RV_MA = get_realized_volatility(Rt_MA)
        RV_MOM = get_realized_volatility(Rt_MOM)

        RV_list = [RV_bh, RV_MA, RV_MOM]
        r_list = [r_bh, r_MA, r_MOM]
        for j in range(3):
            if r_list[j] - norm.ppf(0.99) * RV_list[j] >=0:
                rt_df.iloc[i,j] = str(round(r_list[j] , 4)) + "***"
            elif r_list[j] - norm.ppf(0.95) * RV_list[j] >=0:
                rt_df.iloc[i,j] = str(round(r_list[j] , 4)) + "**"
            elif r_list[j] - norm.ppf(0.9) * RV_list[j] >=0:
                rt_df.iloc[i,j] = str(round(r_list[j] , 4)) + "*"
            else:
                rt_df.iloc[i,j] = str(round(r_list[j] , 4))
        i += 1

print(rt_df.to_latex())


# ================================= Sharpe Ratios ================================= #

qnt = next(iter(quintile_list))
lb = next(iter(lookback_list))
SR_df = pd.DataFrame(columns=["buy-hold", "MA", "TSMOM"],
                       index=list(itertools.product(quintile_list, lookback_list)))

i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=trade_period,
                                                    transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

        RV_bh = get_realized_volatility(Rt_bh, method="annual")
        RV_MA = get_realized_volatility(Rt_MA, method="annual")
        RV_MOM = get_realized_volatility(Rt_MOM, method="annual")

        SR_bh = (r_bh - (tb.DTB3[Rt_bh.index]/100).mean())/RV_bh
        SR_MA = (r_MA - (tb.DTB3[Rt_MA.index]/100).mean())/RV_MA
        SR_MOM = (r_MOM - (tb.DTB3[Rt_MOM.index]/100).mean()).mean()/RV_MOM

        SR_df.iloc[i,:] = [SR_bh, SR_MA, SR_MOM]
        i += 1
print(SR_df.to_latex())


SR_dfs = SR_df.copy()
i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        print("qnt = {} \t lb = {}".format(qnt, lb))
        r_bh_50years = []; r_MA_50years = []; r_MOM_50years = []
        Rt_bh_50years = []; Rt_MA_50years = []; Rt_MOM_50years = []
        SR_bh_list = []; SR_MA_list = []; SR_MOM_list = []
        starts = []
        for y in np.arange(1963, 2014):
            for m in np.arange(1,13):
                period = (datetime(y,m,1), datetime(y,m,1) + relativedelta(months=12))
                Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=period,
                                                          transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
                Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=period,
                                                          transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
                Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=period,
                                                            transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

                r_bh_50years.append(r_bh); r_MA_50years.append(r_MA); r_MOM_50years.append(r_MOM)
                Rt_bh_50years.append(Rt_bh); Rt_MA_50years.append(Rt_MA); Rt_MOM_50years.append(Rt_MOM)
                starts.append(datetime(y,m,1))

                RV_bh = get_realized_volatility(Rt_bh, method="annual")
                RV_MA = get_realized_volatility(Rt_MA, method="annual")
                RV_MOM = get_realized_volatility(Rt_MOM, method="annual")

                SR_bh = (r_bh - (tb.DTB3[Rt_bh.index]/100).mean())/RV_bh
                SR_MA = (r_MA - (tb.DTB3[Rt_MA.index]/100).mean())/RV_MA
                SR_MOM = (r_MOM - (tb.DTB3[Rt_MOM.index]/100).mean()).mean()/RV_MOM

                SR_bh_list.append(SR_bh)
                SR_MA_list.append(SR_MA)
                SR_MOM_list.append(SR_MOM)

        SR_llist = [SR_bh_list, SR_MA_list, SR_MOM_list]
        for j in range(3):
            if SR_dfs.iloc[i,j] - norm.ppf(0.99) * np.std(SR_llist[j]) >=0:
                SR_dfs.iloc[i, j] = str(round(SR_dfs.iloc[i, j], 3)) + "***"
            elif SR_dfs.iloc[i,j] - norm.ppf(0.95) * np.std(SR_llist[j]) >=0:
                SR_dfs.iloc[i, j] = str(round(SR_dfs.iloc[i, j], 3)) + "**"
            elif SR_dfs.iloc[i,j] - norm.ppf(0.9) * np.std(SR_llist[j]) >=0:
                SR_dfs.iloc[i, j] = str(round(SR_dfs.iloc[i, j], 3)) + "*"
            else:
                SR_dfs.iloc[i, j] = str(round(SR_dfs.iloc[i, j], 3))

        i += 1
print(SR_dfs.to_latex())


# ================================= Jensen's Alpha ================================= #

get_jensen_alpha()

qnt = next(iter(quintile_list))
lb = next(iter(lookback_list))
alphas_df = pd.DataFrame(columns=["buy-hold", "MA", "TSMOM"],
                         index=list(itertools.product(quintile_list, lookback_list)))

i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=trade_period,
                                                    transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

        alphas_df.iloc[i,:] = [get_jensen_alpha(Rt_bh), get_jensen_alpha(Rt_MA), get_jensen_alpha(Rt_MOM)]
        i += 1
print(alphas_df.to_latex())


alphas_dfs = alphas_df.copy()
# alphas_dfss = alphas_df.copy()
i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        print("qnt = {} \t lb = {}".format(qnt, lb))
        r_bh_50years = []; r_MA_50years = []; r_MOM_50years = []
        Rt_bh_50years = []; Rt_MA_50years = []; Rt_MOM_50years = []
        alpha_bh_list = []; alpha_MA_list = []; alpha_MOM_list = []
        starts = []
        for y in np.arange(1963, 2014):
            for m in np.arange(1,13):
                period = (datetime(y,m,1), datetime(y,m,1) + relativedelta(months=12))
                Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=period,
                                                          transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
                Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=period,
                                                          transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
                Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=period,
                                                            transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

                r_bh_50years.append(r_bh); r_MA_50years.append(r_MA); r_MOM_50years.append(r_MOM)
                Rt_bh_50years.append(Rt_bh); Rt_MA_50years.append(Rt_MA); Rt_MOM_50years.append(Rt_MOM)
                starts.append(datetime(y,m,1))

                alpha_bh_list.append(get_jensen_alpha(Rt_bh))
                alpha_MA_list.append(get_jensen_alpha(Rt_MA))
                alpha_MOM_list.append(get_jensen_alpha(Rt_MOM))

        alpha_llist = [alpha_bh_list, alpha_MA_list, alpha_MOM_list]
        for j in range(3):
            if np.mean(alpha_llist[j]) - norm.ppf(0.99) * np.std(alpha_llist[j]) >=0:
                alphas_dfs.iloc[i, j] = str(round(np.mean(alpha_llist[j]), 3)) + "***"
            elif np.mean(alpha_llist[j])  - norm.ppf(0.95) * np.std(alpha_llist[j]) >=0:
                alphas_dfs.iloc[i, j] = str(round(np.mean(alpha_llist[j]), 3)) + "**"
            elif np.mean(alpha_llist[j])  - norm.ppf(0.9) * np.std(alpha_llist[j]) >=0:
                alphas_dfs.iloc[i, j] = str(round(np.mean(alpha_llist[j]), 3)) + "*"
            else:
                alphas_dfs.iloc[i, j] = str(round(np.mean(alpha_llist[j]), 3))

            # if alphas_dfs.iloc[i,j] - norm.ppf(0.99) * np.std(alpha_llist[j]) >=0:
            #     alphas_dfs.iloc[i, j] = str(round(alphas_dfs.iloc[i, j], 3)) + "***"
            # elif alphas_dfs.iloc[i,j] - norm.ppf(0.95) * np.std(alpha_llist[j]) >=0:
            #     alphas_dfs.iloc[i, j] = str(round(alphas_dfs.iloc[i, j], 3)) + "**"
            # elif alphas_dfs.iloc[i,j] - norm.ppf(0.9) * np.std(alpha_llist[j]) >=0:
            #     alphas_dfs.iloc[i, j] = str(round(alphas_dfs.iloc[i, j], 3)) + "*"
            # else:
            #     alphas_dfs.iloc[i, j] = str(round(alphas_dfs.iloc[i, j], 3))

        # alphas_dfss.iloc[i,:] = [np.mean(alpha_bh_list), np.mean(alpha_MA_list), np.mean(alpha_MOM_list)]

        i += 1
print(alphas_dfs.to_latex())

# print(alphas_dfss.to_latex())



# ================================= Crash analysis ================================= #


qnt = next(iter(quintile_list))
lb = next(iter(lookback_list))
mkt_timing_df = pd.DataFrame(columns=["buy-hold", "MA", "TSMOM"],
                         index=list(itertools.product(quintile_list, lookback_list)))
mkt_stress_df = pd.DataFrame(columns=["buy-hold", "MA", "TSMOM"],
                         index=list(itertools.product(quintile_list, lookback_list)))

trade_period = (datetime(1963, 1, 1), datetime(1983, 12, 31))  # (datetime(2005,1,1), datetime(2021,12,31)) #

i = 0
for qnt in quintile_list:
    for lb in lookback_list:
        Vt_bh, r_bh, Rt_bh = get_portfolio_value_process(portfolio=qnt, strategy="buy-hold", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MA, r_MA, Rt_MA = get_portfolio_value_process(portfolio=qnt, strategy="MA", trade_period=trade_period,
                                                  transaction_cost=transaction_cost, look_back=lb, return_Rt=True)
        Vt_MOM, r_MOM, Rt_MOM = get_portfolio_value_process(portfolio=qnt, strategy="TSMOM", trade_period=trade_period,
                                                    transaction_cost=transaction_cost, look_back=lb, return_Rt=True)

        Rm_monthly=ff4m["Mkt"] [(ff4m["Mkt"] .index >= trade_period[0]) & (ff4m["Mkt"] .index <= trade_period[1])]
        len(Rm_monthly)
        vol_m = np.array([(ff4.loc[t- timedelta(days=50): t, "Mkt"]).std() for t in Rm_monthly.index])
        I_upmonth = np.array(ff4m.loc[Rm_monthly.index, "Mkt"] > 0).astype(int)

        Rt_list = [Rt_bh, Rt_MA, Rt_MOM]
        for j in range(3):
            Rt_monthly = convert_to_monthly_return(daily_return=Rt_list[j])
            len(Rt_monthly)

            I_bear = np.array([(ff4m.loc[t - relativedelta(months=24): t, "Mkt"]).mean() < 0 for t in Rt_monthly.index]).astype(int)
            # I_bear = np.array([all(ff4m.loc[t - relativedelta(months=5): t, "Mkt"] < 0) for t in Rm_monthly.index]).astype(int)
            len(I_bear)

            df_crash_analysis = ff4m.loc[Rm_monthly.index, :]
            len(df_crash_analysis)
            df_crash_analysis["I_bear"] = I_bear
            df_crash_analysis["I_upmonth"] = I_upmonth
            df_crash_analysis["vol_m"] = vol_m
            df_crash_analysis["Y"] = (Rt_monthly-1)*100 - df_crash_analysis["RF"]
            mkt_timing_formula = "Y ~ I_bear + Mkt_RF + I_bear:Mkt_RF + I_bear:I_upmonth:Mkt_RF"
            lm_mkt_timing = ols(formula=mkt_timing_formula, data=df_crash_analysis).fit()

            p_timing = tuple(round(lm_mkt_timing.params, 3))
            p_timing_str = list(p_timing)
            for z, p in enumerate(p_timing):
                if lm_mkt_timing.pvalues[z] < 0.01:
                    p_timing_str[z] = str(p) + "***"
                elif lm_mkt_timing.pvalues[z] < 0.05:
                    p_timing_str[z] = str(p) + "**"
                elif lm_mkt_timing.pvalues[z] < 0.1:
                    p_timing_str[z] = str(p) + "*"
                else:
                    p_timing_str[z] = str(p)
            mkt_timing_df.iloc[i,j] = tuple(p_timing_str)

            mkt_stress_formula = "Y ~ I_bear + vol_m + I_bear:vol_m"
            lm_mkt_stress = ols(formula=mkt_stress_formula, data=df_crash_analysis).fit()
            p_stress = tuple(round(lm_mkt_stress.params, 3))
            p_stress_str = list(p_stress)
            for z, p in enumerate(p_stress):
                if lm_mkt_stress.pvalues[z] < 0.01:
                    p_stress_str[z] = str(p) + "***"
                elif lm_mkt_stress.pvalues[z] < 0.05:
                    p_stress_str[z] = str(p) + "**"
                elif lm_mkt_stress.pvalues[z] < 0.1:
                    p_stress_str[z] = str(p) + "*"
                else:
                    p_stress_str[z] = str(p)
            mkt_stress_df.iloc[i, j] = tuple(p_stress_str)

        i +=1

print(mkt_timing_df.iloc[:,1:].to_latex())

print(mkt_stress_df.iloc[:,1:].to_latex())


# mkt_timing_formula = "Y ~ I_bear + Mkt_RF + I_bear:Mkt_RF + I_bear:I_upmonth:Mkt_RF"
# lm_mkt_timing = ols(formula=mkt_timing_formula, data=df_crash_analysis).fit()
# lm_mkt_timing.summary()
#
#
# mkt_stress_formula = "Y ~ I_bear + vol_m + I_bear:vol_m"
# lm_mkt_stress = ols(formula=mkt_stress_formula, data=df_crash_analysis).fit()
# lm_mkt_stress.summary()





