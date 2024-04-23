import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class PairTradingService:
    def __init__(self, std_dev_factor=1.5):
        self.std_dev_factor = std_dev_factor
        self.portfolio_history = []

    def find_pairs(self, cluster_data):
        cluster_data = cluster_data.reset_index()
        sorted_data = cluster_data.sort_values(by='mom1m', ascending=False)
        pairs = []

        for i in range(len(sorted_data) // 2):
            top_stock = sorted_data.iloc[i]
            bottom_stock = sorted_data.iloc[-(i + 1)]
            diff = top_stock['mom1m'] - bottom_stock['mom1m']
            pairs.append((top_stock, bottom_stock, diff))

        std_dev = np.std([p[2] for p in pairs])
        valid_pairs = [p for p in pairs if p[2] > std_dev * self.std_dev_factor]

        return valid_pairs

    def form_portfolio(self, pairs):
        long_short_portfolio = {'long': [], 'short': []}
        for top_stock, bottom_stock, _ in pairs:
            long_short_portfolio['long'].append(bottom_stock['permno'])
            long_short_portfolio['short'].append(top_stock['permno'])
        return long_short_portfolio

    def manage_monthly_trades(self, grouped_data):
        month_data_dict = {}
        portfolios_dict = {}
        month_return = []
        total_return = 1

        # tqdm is used here to show the progress of managing monthly trades
        for month, month_data in tqdm(grouped_data, desc="Managing monthly trades"):
            month_data_dict[month] = month_data

        for month, month_data in tqdm(month_data_dict.items(), desc="Processing each month"):
            grouped_by_cluster = month_data.groupby('cluster')
            current_month_portfolio = {'long': [], 'short': []}

            for cluster, cluster_data in grouped_by_cluster:
                pairs = self.find_pairs(cluster_data)
                portfolio = self.form_portfolio(pairs)
                current_month_portfolio['long'].extend(portfolio['long'])
                current_month_portfolio['short'].extend(portfolio['short'])

            portfolios_dict[month] = current_month_portfolio

            next_month = month + pd.DateOffset(months=1)
            if next_month in month_data_dict:
                next_month_data = month_data_dict[next_month]
                returns = self.calculate_monthly_returns(current_month_portfolio, next_month_data)
                month_return.append([month, returns])

        for data in tqdm(month_return, desc="Calculating total returns"):
            total_return *= (1 + data[1])

        return total_return, month_return

    def calculate_monthly_returns(self, portfolio, month_data):
        long_returns = []
        short_returns = []
        month_data.set_index('permno', inplace=True)

        for permno in portfolio['long']:
            if permno in month_data.index:
                long_returns.append(month_data.loc[permno, 'mom1m'])

        for permno in portfolio['short']:
            if permno in month_data.index:
                short_returns.append(-month_data.loc[permno, 'mom1m'])

        total_returns = long_returns + short_returns
        average_return = np.mean(total_returns) if total_returns else 0
        return average_return