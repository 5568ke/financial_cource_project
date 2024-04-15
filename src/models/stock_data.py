import pandas as pd

class FeatureData:
    def __init__(self, price,permno, date, mvel1, beta, betasq, chmom, dolvol, idiovol, indmom,
                 mom1m, mom6m, mom12m, mom36m, pricedelay, turn, absacc, acc, age, agr,
                 bm, bm_ia, cashdebt, cashpr, cfp, cfp_ia, chatoia, chcsho, chempia, chinv,
                 chpmia, convind, currat, depr, divi, divo, dy, egr, ep, gma, grcapx, grltnoa,
                 herf, hire, invest, lev, lgr, mve_ia, operprof, orgcap, pchcapx_ia, pchcurrat,
                 pchdepr, pchgm_pchsale, pchquick, pchsale_pchinvt, pchsale_pchrect, pchsale_pchxsga,
                 pchsaleinv, pctacc, ps, quick, rd, rd_mve, rd_sale, realestate, roic, salecash,
                 saleinv, salerec, secured, securedind, sgr, sin, sp, tang, tb, aeavol, cash, chtx,
                 cinvest, ear, nincr, roaq, roavol, roeq, rsup, stdacc, stdcf, ms, baspread, ill,
                 maxret, retvol, std_dolvol, std_turn, zerotrade, sic2):
        self.price = price
        self.permno = permno
        self.date = date
        self.mvel1 = mvel1
        self.beta = beta
        self.betasq = betasq
        self.chmom = chmom
        self.dolvol = dolvol
        self.idiovol = idiovol
        self.indmom = indmom
        self.mom1m = mom1m
        self.mom6m = mom6m
        self.mom12m = mom12m
        self.mom36m = mom36m
        self.pricedelay = pricedelay
        self.turn = turn
        self.absacc = absacc
        self.acc = acc
        self.age = age
        self.agr = agr
        self.bm = bm
        self.bm_ia = bm_ia
        self.cashdebt = cashdebt
        self.cashpr = cashpr
        self.cfp = cfp
        self.cfp_ia = cfp_ia
        self.chatoia = chatoia
        self.chcsho = chcsho
        self.chempia = chempia
        self.chinv = chinv
        self.chpmia = chpmia
        self.convind = convind
        self.currat = currat
        self.depr = depr
        self.divi = divi
        self.divo = divo
        self.dy = dy
        self.egr = egr
        self.ep = ep
        self.gma = gma
        self.grcapx = grcapx
        self.grltnoa = grltnoa
        self.herf = herf
        self.hire = hire
        self.invest = invest
        self.lev = lev
        self.lgr = lgr
        self.mve_ia = mve_ia
        self.operprof = operprof
        self.orgcap = orgcap
        self.pchcapx_ia = pchcapx_ia
        self.pchcurrat = pchcurrat
        self.pchdepr = pchdepr
        self.pchgm_pchsale = pchgm_pchsale
        self.pchquick = pchquick
        self.pchsale_pchinvt = pchsale_pchinvt
        self.pchsale_pchrect = pchsale_pchrect
        self.pchsale_pchxsga = pchsale_pchxsga
        self.pchsaleinv = pchsaleinv
        self.pctacc = pctacc
        self.ps = ps
        self.quick = quick
        self.rd = rd
        self.rd_mve = rd_mve
        self.rd_sale = rd_sale
        self.realestate = realestate
        self.roic = roic
        self.salecash = salecash
        self.saleinv = saleinv
        self.salerec = salerec
        self.secured = secured
        self.securedind = securedind
        self.sgr = sgr
        self.sin = sin
        self.sp = sp
        self.tang = tang
        self.tb = tb
        self.aeavol = aeavol
        self.cash = cash
        self.chtx = chtx
        self.cinvest = cinvest
        self.ear = ear
        self.nincr = nincr
        self.roaq = roaq
        self.roavol = roavol
        self.roeq = roeq
        self.rsup = rsup
        self.stdacc = stdacc
        self.stdcf = stdcf
        self.ms = ms
        self.baspread = baspread
        self.ill = ill
        self.maxret = maxret
        self.retvol = retvol
        self.std_dolvol = std_dolvol
        self.std_turn = std_turn
        self.zerotrade = zerotrade
        self.sic2 = sic2



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

