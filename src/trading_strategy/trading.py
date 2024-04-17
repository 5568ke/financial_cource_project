import pandas as pd
import numpy as np

class PairTradingService:
    def __init__(self, std_dev_factor=1.5):
        self.std_dev_factor = std_dev_factor
        self.portfolio_history = []

    def find_pairs(self, cluster_data):
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

    def execute_trades(self, portfolio):
        print(f"Going long on: {portfolio['long']}")
        print(f"Going short on: {portfolio['short']}")


    def calculate_monthly_returns(self, portfolio, month_data):

        long_returns = []
        short_returns = []
        month_data.set_index('permno', inplace=True)
        print("month_data : ",month_data["mom1m"])
        for permno in portfolio['long']:
            if permno in month_data.index:
                long_returns.append(month_data.loc[permno, 'mom1m'])

        for permno in portfolio['short']:
            if permno in month_data.index:
                short_returns.append(-month_data.loc[permno, 'mom1m'])

        total_returns = long_returns + short_returns
        average_return = np.mean(total_returns) if total_returns else 0
        self.portfolio_history.append({
            'portfolio': portfolio,
            'return': average_return
        })
        return average_return
