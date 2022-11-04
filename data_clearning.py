import numpy as np
import pandas as pd

data_PATH = "/Users/y222chen/Documents/Max/Study/ACTSC972-Finance3/Project/data/"

# df_full_daily = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=12)
# pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=11, nrows=2, delim_whitespace=True)
# df_full_daily.shape


value_weighted_return_daily = pd.read_csv(data_PATH+"Portfolios_Formed_on_ME_daily.csv", skiprows=12, nrows=25336)
column_names = pd.Index(np.append(["Date"],list(value_weighted_return_daily.columns[1:])))
value_weighted_return_daily.columns = column_names
value_weighted_return_daily.Date = pd.to_datetime(value_weighted_return_daily.Date, format="%Y%m%d")

value_weighted_return_daily.shape #[25336, 20]0.
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


value_weighted_quintile_portfolio_returns = value_weighted_return_daily[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]

equal_weighted_quintile_portfolio_returns = equal_weighted_return_daily[["Date", "Lo 20", 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20',]]

print("Both value_weighted and equal_weighted quintile portfolios are successfully generated.")


tbill3m_daily = pd.read_csv(data_PATH+"TB3MS.csv")
tbill3m_daily.head()






