import pandas as pd
from models.stock_data import Stock, FeatureData

class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        total_rows = len(data)
        stocks = {}
        print_interval = 10000 

        for index, row in data.iterrows():
            permno = row['permno']

            if permno not in stocks:
                stocks[permno] = Stock(permno)

            feature_data = FeatureData(
                price=100,  
                permno=row['permno'], date=row['DATE'], mvel1=row['mvel1'], beta=row['beta'],
                betasq=row['betasq'], chmom=row['chmom'], dolvol=row['dolvol'], idiovol=row['idiovol'],
                indmom=row['indmom'], mom1m=row['mom1m'], mom6m=row['mom6m'], mom12m=row['mom12m'],
                mom36m=row['mom36m'], pricedelay=row['pricedelay'], turn=row['turn'], absacc=row['absacc'], 
                acc=row['acc'], age=row['age'], agr=row['agr'], bm=row['bm'], bm_ia=row['bm_ia'], 
                cashdebt=row['cashdebt'], cashpr=row['cashpr'], cfp=row['cfp'], cfp_ia=row['cfp_ia'],
                chatoia=row['chatoia'], chcsho=row['chcsho'], chempia=row['chempia'], chinv=row['chinv'],
                chpmia=row['chpmia'], convind=row['convind'], currat=row['currat'], depr=row['depr'],
                divi=row['divi'], divo=row['divo'], dy=row['dy'], egr=row['egr'], ep=row['ep'], gma=row['gma'],
                grcapx=row['grcapx'], grltnoa=row['grltnoa'], herf=row['herf'], hire=row['hire'],
                invest=row['invest'], lev=row['lev'], lgr=row['lgr'], mve_ia=row['mve_ia'], operprof=row['operprof'],
                orgcap=row['orgcap'], pchcapx_ia=row['pchcapx_ia'], pchcurrat=row['pchcurrat'],
                pchdepr=row['pchdepr'], pchgm_pchsale=row['pchgm_pchsale'], pchquick=row['pchquick'],
                pchsale_pchinvt=row['pchsale_pchinvt'], pchsale_pchrect=row['pchsale_pchrect'],
                pchsale_pchxsga=row['pchsale_pchxsga'], pchsaleinv=row['pchsaleinv'], pctacc=row['pctacc'],
                ps=row['ps'], quick=row['quick'], rd=row['rd'], rd_mve=row['rd_mve'], rd_sale=row['rd_sale'],
                realestate=row['realestate'], roic=row['roic'], salecash=row['salecash'], saleinv=row['saleinv'],
                salerec=row['salerec'], secured=row['secured'], securedind=row['securedind'], sgr=row['sgr'],
                sin=row['sin'], sp=row['sp'], tang=row['tang'], tb=row['tb'], aeavol=row['aeavol'], cash=row['cash'],
                chtx=row['chtx'], cinvest=row['cinvest'], ear=row['ear'], nincr=row['nincr'], roaq=row['roaq'],
                roavol=row['roavol'], roeq=row['roeq'], rsup=row['rsup'], stdacc=row['stdacc'], stdcf=row['stdcf'],
                ms=row['ms'], baspread=row['baspread'], ill=row['ill'], maxret=row['maxret'], retvol=row['retvol'],
                std_dolvol=row['std_dolvol'], std_turn=row['std_turn'], zerotrade=row['zerotrade'], sic2=row['sic2']
            )
            stocks[permno].add_month_data(date=row['DATE'], feature_data=feature_data)

            if (index + 1) % print_interval == 0 or index + 1 == total_rows:
                print(f"Processed {index + 1} out of {total_rows} rows ({(index + 1) / total_rows * 100:.2f}%)")

        return stocks
