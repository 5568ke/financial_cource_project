class FeatureData:
    FEATURE_NAMES = [
        'permno', 'date', 'mvel1', 'beta', 'betasq', 'chmom', 'dolvol', 'idiovol',
        'indmom', 'mom1m', 'mom6m', 'mom12m', 'mom36m', 'pricedelay', 'turn', 
        'absacc', 'acc', 'age', 'agr', 'bm', 'bm_ia', 'cashdebt', 'cashpr', 'cfp', 
        'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv', 'chpmia', 'convind', 
        'currat', 'depr', 'divi', 'divo', 'dy', 'egr', 'ep', 'gma', 'grcapx', 
        'grltnoa', 'herf', 'hire', 'invest', 'lev', 'lgr', 'mve_ia', 'operprof', 
        'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick', 
        'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 'pchsaleinv', 
        'pctacc', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale', 'realestate', 'roic', 
        'salecash', 'saleinv', 'salerec', 'secured', 'securedind', 'sgr', 'sin', 
        'sp', 'tang', 'tb', 'aeavol', 'cash', 'chtx', 'cinvest', 'ear', 'nincr', 
        'roaq', 'roavol', 'roeq', 'rsup', 'stdacc', 'stdcf', 'ms', 'baspread', 
        'ill', 'maxret', 'retvol', 'std_dolvol', 'std_turn', 'zerotrade', 'sic2'
    ]

    
    def __init__(self, *args):
        self.features = list(args)
    
    def get_feature(self, feature_name):
        index = self.FEATURE_NAMES.index(feature_name)
        return self.features[index] if index < len(self.features) else None

    def set_feature(self, feature_name, value):
        index = self.FEATURE_NAMES.index(feature_name)
        if index < len(self.features):
            self.features[index] = value
        else:
            while len(self.features) <= index:
                self.features.append(None)
            self.features[index] = value

class Stock:
    def __init__(self, permno):
        self.permno = permno
        self.monthly_data = {}
    
    def add_month_data(self, date, feature_data):
        self.monthly_data[date] = feature_data
    
    def calculate_momentum(self, date, months=1):
        pass
    
    def get_month_data(self, date):
        return self.monthly_data.get(date, None)
