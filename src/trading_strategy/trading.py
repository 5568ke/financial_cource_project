import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import config
from config import PATHS, STRESS_TEST_PERIODS
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

        #std_dev = np.std([p[2] for p in pairs])
        #valid_pairs = [p for p in pairs if p[2] > std_dev * self.std_dev_factor]

        return pairs

    def form_portfolio(self, pairs):
        long_short_portfolio = {'long': [], 'short': []}
        for top_stock, bottom_stock, _ in pairs:
            long_short_portfolio['long'].append(bottom_stock['permno'])
            long_short_portfolio['short'].append(top_stock['permno'])
        return long_short_portfolio

    def process_month(self,month, month_data):
        grouped_by_cluster = month_data.groupby('cluster')
        current_month_portfolio = {'long': [], 'short': []}

        all_pairs=[]
        for cluster, cluster_data in grouped_by_cluster:
            pairs = self.find_pairs(cluster_data)
            all_pairs += pairs

        std_dev = np.std([p[2] for p in all_pairs])
        valid_pairs = [p for p in all_pairs if p[2] > std_dev * self.std_dev_factor]

        #print(" pair data for month ", month, "\n pair numbers: ", len(valid_pairs))

        portfolio = self.form_portfolio(valid_pairs)
        current_month_portfolio['long'].extend(portfolio['long'])
        current_month_portfolio['short'].extend(portfolio['short'])

        return month, current_month_portfolio

    def manage_monthly_trades(self, grouped_data, suffix="full_period"):
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

        total_returns = [1]
        monthly_returns = []
        for i, data in enumerate(tqdm(month_return, desc="Calculating total returns")):
            total_return *= (1 + data[1])
            monthly_returns.append(data[1])
            if suffix != "full_period" or (i + 1) % 12 == 0:
                total_returns.append(total_return)

        self.save_asset_changes_plot(total_returns,suffix)
        self.save_monthly_returns_plot(monthly_returns,suffix)

        # Calculate IRR
        if suffix == "full_period":
            print("Annual IRR : ", total_return ** (1 / len(total_returns))*100-100,"%")
        else:
            print("Annual IRR : ", total_return ** (12 / len(total_returns))*100-100,"%")
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
        return average_return * 0.9985 # from https://www.sec.gov/edgar/filer/filing-fees/filing-fee-rate

    def save_asset_changes_plot(self, total_returns, suffix):
        """
        Save the plot of asset changes over the investment period.

        Args:
            total_returns (list): List of total returns over the investment period.
        """
        plot_path = PATHS[config.CLUSTERING_ALGORITHM]['plot'].replace(".png", f"_{suffix}.png")  # Get the plot path from the config
        plt.figure(figsize=(10, 6))
        plt.plot(total_returns, marker='o', linestyle='-')
        plt.title(f'Asset Changes Over Investment Period ({suffix})')
        plt.xlabel('Years')
        plt.ylabel('Total Return')
        plt.grid(True)
        plt.savefig(plot_path)  # Save the plot to the specified path'

        # Store total_returns data
        data_folder = PATHS[config.CLUSTERING_ALGORITHM]['returns_list'].replace("_full_period.pkl", f"_{suffix}.pkl")  # Get the data folder path from the config
        total_returns_file = os.path.join(data_folder)  # Path to store total_returns data
        with open(total_returns_file, 'wb') as f:
            pickle.dump(total_returns, f)

    def save_monthly_returns_plot(self, monthly_returns,suffix):
        """
        Save the plot of monthly returns over the investment period.

        Args:
            monthly_returns (list): List of monthly returns over the investment period.
        """
        #print(monthly_returns)
        plot_path = PATHS[config.CLUSTERING_ALGORITHM]['monthly_return_plot'].replace(".png", f"_{suffix}.png")  # Get the plot path from the config
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_returns, marker='o', linestyle='-')
        plt.title(f'Monthly Returns Over Investment Period ({suffix})')
        plt.xlabel('Months')
        plt.ylabel('Monthly Return')
        plt.grid(True)
        plt.savefig(plot_path)  # Save the plot to the specified path

    def run_stress_tests(self, clustered_data):
        for period in STRESS_TEST_PERIODS:
            start_date = pd.to_datetime(period["start"])
            end_date = pd.to_datetime(period["end"])
            stress_data = clustered_data.loc[start_date:end_date].copy()
            grouped_by_month = stress_data.groupby(pd.Grouper(freq='M'))
            print(f"Running stress test for period: {start_date.date()} to {end_date.date()}")
            self.manage_monthly_trades(grouped_by_month, suffix=f"{start_date.date()}_to_{end_date.date()}")
