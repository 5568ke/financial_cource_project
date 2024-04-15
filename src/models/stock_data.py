import pandas as pd

class FeatureData:
    def __init__(self, price, **features):
        self.price = price
        self.features = features

class Stock:
    def __init__(self, permno):
        self.permno = permno
        self.monthly_data = {}
    
    def add_month_data(self, date, price, **features):
        self.monthly_data[date] = FeatureData(price, **features)
    
    def calculate_momentum(self, date, months=1):
    
    def get_month_data(self, date):
        return self.monthly_data.get(date, None)

