import numpy as np
import pandas as pd
import pandas.tseries.offsets as pdoffsets
# import torch as tc
# tc.backends.mps.is_available()


# data_PATH = "/Users/y222chen/Documents/Max/Study/ACTSC972-Finance3/Project/data/"
data_PATH = "/Users/maxchen/Documents/Study/STA/ACTSC972-Finance3/Project/project/data/"


# df_full_daily = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=12)
# pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=11, nrows=2, delim_whitespace=True)
# df_full_daily.shape


value_weighted_return_daily = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=12, nrows=25336)
column_names = pd.Index(np.append(["Date"],list(value_weighted_return_daily.columns[1:])))
value_weighted_return_daily.columns = column_names
value_weighted_return_daily.Date = pd.to_datetime(value_weighted_return_daily.Date, format="%Y%m%d")

value_weighted_return_daily.shape #[25336, 20].
value_weighted_return_daily.head()

# colnames_value_weighted = tuple(zip(["Value_Weighted_Returns_Daily"]*len(list(l2_column_names)), list(l2_column_names)))
# header_value_weighted = pd.MultiIndex.from_tuples(colnames_value_weighted)
# df_daily.columns = header_value_weighted
# df_daily.loc[0]["Value_Weighted_Returns_Daily", "<= 0"]


equal_weighted_return_daily = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=25336+16)
equal_weighted_return_daily.columns = column_names
equal_weighted_return_daily.Date = pd.to_datetime(equal_weighted_return_daily.Date, format="%Y%m%d")
# equal_weighted_return_daily.head()
equal_weighted_return_daily.shape #[25336, 20]


value_weighted_quintile_portfolio_returns_daily = value_weighted_return_daily[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]
value_weighted_quintile_portfolio_returns_daily = value_weighted_quintile_portfolio_returns_daily.set_index(keys="Date")

equal_weighted_quintile_portfolio_returns_daily = equal_weighted_return_daily[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]
equal_weighted_quintile_portfolio_returns_daily = equal_weighted_quintile_portfolio_returns_daily.set_index("Date")
print("Both value_weighted and equal_weighted quintile portfolios (daily) are successfully generated.")


tbill3m_daily = pd.read_csv(data_PATH+"DTB3.csv")
tbill3m_daily.DATE = pd.to_datetime(tbill3m_daily.DATE)
tbill3m_daily.DTB3 = tbill3m_daily.DTB3.replace(".", np.nan).astype("float")
tbill3m_daily.DTB3.interpolate(method = "linear", inplace = True)
tbill3m_daily.head()
tbill3m_daily = tbill3m_daily.set_index(keys="DATE")
print("3 month Tbill secondary market rates (daily frequency) are successfully generated.")

value_weighted_return_monthly = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME.csv", skiprows=12, nrows=1168-13)
value_weighted_return_monthly.columns = column_names
value_weighted_return_monthly.Date = pd.to_datetime(value_weighted_return_monthly.Date, format="%Y%m") + pdoffsets.MonthEnd(0)
value_weighted_quintile_portfolio_returns_monthly = value_weighted_return_monthly[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]
value_weighted_quintile_portfolio_returns_monthly = value_weighted_quintile_portfolio_returns_monthly.set_index(keys="Date")

equal_weighted_return_monthly = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME.csv", skiprows=1171, nrows=2327-1172)
equal_weighted_return_monthly.columns = column_names
equal_weighted_return_monthly.Date = pd.to_datetime(equal_weighted_return_monthly.Date, format="%Y%m") + pdoffsets.MonthEnd(0)
equal_weighted_quintile_portfolio_returns_monthly = equal_weighted_return_monthly[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]
equal_weighted_quintile_portfolio_returns_monthly = equal_weighted_quintile_portfolio_returns_monthly.set_index(keys="Date")
print("Both value_weighted and equal_weighted quintile portfolios (monthly) are successfully generated.")

value_weighted_return_annual = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME.csv", skiprows=2330, nrows=2426-2331)
value_weighted_return_annual.columns = column_names
value_weighted_return_annual.Date = pd.to_datetime(value_weighted_return_annual.Date, format="%Y") + pdoffsets.YearEnd(0)
value_weighted_quintile_portfolio_returns_annual = value_weighted_return_annual[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]
value_weighted_quintile_portfolio_returns_annual = value_weighted_quintile_portfolio_returns_annual.set_index(keys="Date")

equal_weighted_return_annual = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME.csv", skiprows=2429, nrows=2525-2430)
equal_weighted_return_annual.columns = column_names
equal_weighted_return_annual.Date = pd.to_datetime(equal_weighted_return_annual.Date, format="%Y") + pdoffsets.YearEnd(0)
equal_weighted_quintile_portfolio_returns_annual = equal_weighted_return_annual[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]
equal_weighted_quintile_portfolio_returns_annual = (equal_weighted_quintile_portfolio_returns_annual.set_index(keys="Date"))
print("Both value_weighted and equal_weighted quintile portfolios (annual) are successfully generated.")


fama_french_4factors_daily = pd.read_csv(data_PATH + "F-F_Research_Data_Factors_daily.csv", skiprows=4)
fama_french_4factors_daily = fama_french_4factors_daily.iloc[:-1,:]
column_names = pd.Index(np.append(["Date"],list(fama_french_4factors_daily.columns[1:])))
fama_french_4factors_daily.columns = column_names
fama_french_4factors_daily.Date = pd.to_datetime(fama_french_4factors_daily.Date, format="%Y%m%d")
fama_french_4factors_daily = fama_french_4factors_daily.set_index(keys="Date")
fama_french_4factors_daily = fama_french_4factors_daily.rename(columns = {"Mkt-RF": "Mkt_RF"})
fama_french_4factors_daily["Mkt"] = fama_french_4factors_daily["Mkt_RF"] + fama_french_4factors_daily["RF"]
print("Fama French 4 factors daily is succesfully generated.")

fama_french_4factors_monthly = pd.read_csv(data_PATH+"F-F_Research_Data_Factors.CSV", skiprows=3, nrows=1159-4)
column_names = pd.Index(np.append(["Date"],list(fama_french_4factors_monthly.columns[1:])))
fama_french_4factors_monthly.columns = column_names
fama_french_4factors_monthly.Date = pd.to_datetime(fama_french_4factors_monthly.Date, format="%Y%m")
fama_french_4factors_monthly = fama_french_4factors_monthly.set_index(keys="Date")
fama_french_4factors_monthly = fama_french_4factors_monthly.rename(columns = {"Mkt-RF": "Mkt_RF"})
fama_french_4factors_monthly["Mkt"] = fama_french_4factors_monthly["Mkt_RF"] + fama_french_4factors_monthly["RF"]
print("Fama French 4 factors monthly is succesfully generated.")







