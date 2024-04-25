import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import config
from config import PATHS
import matplotlib.pyplot as plt
import pickle
import os
import concurrent.futures

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

    def process_month(self,month, month_data):
        grouped_by_cluster = month_data.groupby('cluster')
        current_month_portfolio = {'long': [], 'short': []}

        for cluster, cluster_data in grouped_by_cluster:
            pairs = self.find_pairs(cluster_data)
            portfolio = self.form_portfolio(pairs)
            current_month_portfolio['long'].extend(portfolio['long'])
            current_month_portfolio['short'].extend(portfolio['short'])

        return month, current_month_portfolio

    def manage_monthly_trades(self, grouped_data):
        month_data_dict = {}
        portfolios_dict = {}
        month_return = []
        total_return = 1

        # tqdm is used here to show the progress of managing monthly trades
        for month, month_data in tqdm(grouped_data, desc="Managing monthly trades"):
            month_data_dict[month] = month_data

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_month, month, month_data) for month, month_data in
                       month_data_dict.items()]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing trade of each month"):
                month, current_month_portfolio = future.result()
                portfolios_dict[month] = current_month_portfolio

                next_month = month + pd.DateOffset(months=1)
                next_month_key = next(
                    (m for m in month_data_dict.keys() if m.year == next_month.year and m.month == next_month.month),
                    None)
                if next_month_key:
                    next_month_data = month_data_dict[next_month_key]
                    returns = self.calculate_monthly_returns(current_month_portfolio, next_month_data)
                    month_return.append([month, returns])

        total_returns=[]
        for i, data in enumerate(tqdm(month_return, desc="Calculating total returns")):
            total_return *= (1 + data[1])
            if (i + 1) % 12 == 0:
                total_returns.append(total_return)

        self.save_asset_changes_plot(total_returns)

        # Calculate IRR
        print("Annual IRR : ", total_return ** (1 / len(total_returns))*100-100,"%")
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

    def save_asset_changes_plot(self,total_returns):
        """
        Save the plot of asset changes over the investment period.

        Args:
            total_returns (list): List of total returns over the investment period.
        """
        plot_path = PATHS[config.CLUSTERING_ALGORITHM]['plot']  # Get the plot path from the config
        plt.figure(figsize=(10, 6))
        plt.plot(total_returns, marker='o', linestyle='-')
        plt.title('Asset Changes Over Investment Period')
        plt.xlabel('Years')
        plt.ylabel('Total Return')
        plt.grid(True)
        plt.savefig(plot_path)  # Save the plot to the specified path'

        # Store total_returns data
        data_folder = PATHS[config.CLUSTERING_ALGORITHM]['returns_list']  # Get the data folder path from the config
        total_returns_file = os.path.join(data_folder, '..', 'total_returns.pkl')  # Path to store total_returns data
        with open(total_returns_file, 'wb') as f:
            pickle.dump(total_returns, f)