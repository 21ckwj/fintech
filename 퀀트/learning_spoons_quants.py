# -*- coding: utf-8 -*-
import pandas as pd



# 재무데이터 전처리 해서 가져오기
def finance_data_preprocessing(path):
    
    raw_data = pd.read_excel(path)

    big_col = list(raw_data.columns)
    for num, temp in enumerate(big_col):
        if '.' in temp:
            big_col[num] = temp.split('.')[0]

    small_col = list(raw_data.loc[0])
    small_col[0] = big_col[0]
    small_col[1] = big_col[1]
    
    raw_data.columns = [big_col, small_col]
    raw_data = raw_data.drop(0)
    
    raw_data.index = raw_data['Symbol']['Symbol']
    del raw_data['Symbol']
    
    return raw_data


# 가격데이터 전처리 해서 가져오기
def price_data_preprocs(path):
    price_raw = pd.read_csv(path, engine='python')
    price_df = price_raw.drop(0)
    price_df.index = pd.to_datetime(price_df['코드명'])
    del price_df['코드명']
    return price_df


def grahum_last_present(cleaned_data , date, num):
    cleaned_data = cleaned_data[[('Symbol Name', 'Symbol Name'), ('ROA', date), ('부채비율', date),('PBR', date)]]
    roa5 = cleaned_data[cleaned_data['ROA'][date] > 5]
    roa5deb50 = roa5 [roa5['부채비율'][date] <= 50]
    pbr_filter = roa5deb50[roa5deb50['PBR'][date] > 0.2]
    grahum_last_present = pbr_filter .sort_values(by=('PBR', date))
    return grahum_last_present[:num]


def NCAV(cleaned_data, date, num):
    date2 = str(int(date.split('/')[0])) + '년'
    cleaned_data = cleaned_data[[('Symbol Name', 'Symbol Name'), ('부채비율', date), ('내돈', date), ('유동자산', date2), ('시총', date), ('순익', date)]]
    total_debt = cleaned_data['부채비율'][date]/100 * cleaned_data['내돈'][date]
    temp_result = cleaned_data[ cleaned_data['유동자산'][date2] - total_debt > cleaned_data['시총'][date] * 1.5 ]
    result = temp_result[temp_result['순익'][date] > 0].sort_values(by=('순익', date), ascending=False)
    return result[:num]


def small_low_pbr(cleaned_data, date, num):
    cleaned_data = cleaned_data[[('Symbol Name', 'Symbol Name'), ('시총', date), ('PBR', date)]]
    filter1 = cleaned_data [ cleaned_data['시총'][date] > 0 ]
    low20 = int(len(filter1) * 0.2)
    low_data = filter1.sort_values(by=('시총', date))[:low20]
    low_pbr = low_data[low_data['PBR'][date] > 0.2].sort_values(by=('PBR', date))
    return low_pbr[:num]


def Fama_LSV(cleaned_data, date, num):
    cleaned_data = cleaned_data[[('Symbol Name', 'Symbol Name'), ('시총', date), ('PBR', date), ('PER', date)]]
    positive_ta = cleaned_data[ cleaned_data['시총'][date] > 0 ]
    positve500 = positive_ta.sort_values(by=('시총', date))[:500]
    positve500['PER랭크'] = positve500[positve500['PER'][date] > 0]['PER'][date].rank()
    positve500['PBR랭크'] = positve500[positve500['PBR'][date] > 0]['PBR'][date].rank()
    positve500['total랭크'] = (positve500['PER랭크'] + positve500['PBR랭크']).rank()
    fama_lsv_result = positve500.sort_values(by='total랭크')
    return fama_lsv_result[:num]


def Kang_super_value(cleaned_data, date, num):
    cleaned_data = cleaned_data[[('Symbol Name', 'Symbol Name'), ('시총', date), ('PER', date), ('PBR', date), ('PSR', date)]]
    positive_ta = cleaned_data[cleaned_data['시총'][date] > 0]
    sorted_pta = positive_ta.sort_values(by=('시총', date))
    low_ta = sorted_pta[:int(len(sorted_pta)*0.2)]
    positive_per = low_ta[low_ta['PER'][date] > 0]
    positive_per[('per랭크', date)] = positive_per['PER'][date].rank()
    positive_pbr = positive_per[positive_per['PBR'][date] > 0]
    positive_pbr[('pbr랭크', date)] = positive_pbr['PBR'][date].rank()
    positive_psr = positive_pbr[positive_pbr['PSR'][date] > 0]
    positive_psr[('psr랭크', date)] = positive_psr['PSR'][date].rank()
    positive_psr[('total', date)] = (positive_psr['per랭크'][date] + positive_psr['pbr랭크'][date] +  positive_psr['psr랭크'][date]).rank()
    result = positive_psr.sort_values(by=('total', date))
    return result[:num]


def New_Magic(finance_data, index_date, num):
    year_date = str(int(index_date.split('/')[0])) + '년'
    finance_data = finance_data[[('Symbol Name', 'Symbol Name'), ('총자산', year_date), ('매출총이익율', index_date), ('매출', index_date), ('시총', index_date), ('PBR', index_date)]]
    filt_ta = finance_data[finance_data['총자산'][year_date] > 0]
    filt_ta['GPA'] = (filt_ta['매출총이익율'][index_date] * filt_ta['매출'][index_date])/ filt_ta['총자산'][year_date]
    filt_ta2 = filt_ta[filt_ta['시총'][index_date] > 0]
    low20 = int(len(filt_ta2) * 0.2)
    filt_tp = filt_ta2.sort_values(by=('시총', index_date))[:low20]
    filt_tp['PBR랭크'] = filt_tp[filt_tp['PBR'][index_date] > 0]['PBR'][index_date].rank()
    filt_tp['GPA랭크'] = filt_tp['GPA'].rank(ascending=False)
    filt_tp['종합순위'] = (filt_tp['PBR랭크'] + filt_tp['GPA랭크']).rank()
    new_magic = filt_tp.sort_values(by='종합순위')
    return new_magic[:num]


def F_score_Low_PBR(finance_data, index_date, num):
    finance_data = finance_data[[('Symbol Name', 'Symbol Name'), ('PBR', index_date), ('영업활동현흐', index_date), ('순익', index_date)]]
    filt_pbr = finance_data[finance_data['PBR'][index_date] > 0]
    low_20 = int(len(filt_pbr)*0.2)
    pbr_low20 = filt_pbr.sort_values(by=('PBR', index_date))[:low_20]
    pbr_low20['영흐점수'] = pbr_low20['영업활동현흐'][index_date] > 0
    pbr_low20['순익점수'] = pbr_low20['순익'][index_date] > 0
    pbr_low20['F-score'] = pbr_low20[['순익점수', '영흐점수']].sum(axis=1)
    f_score = pbr_low20[pbr_low20['F-score'] == 2]
    f_score_pbr = f_score.sort_values(by=('PBR', index_date))
    return f_score_pbr[:num]


def Kang_Super_Quality(finance_data, index_date, num):
    year_date = str(int(index_date.split('/')[0])) + '년'
    finance_data = finance_data[[('Symbol Name', 'Symbol Name'), ('매출총이익율', index_date), ('매출', index_date), ('총자산', year_date), ('영업활동현흐', index_date), ('순익', index_date), ('시총', index_date)]]
    filt_ta = finance_data[finance_data['총자산'][year_date] > 0]
    filt_ta['GPA'] = (filt_ta['매출총이익율'][index_date] * filt_ta['매출'][index_date])/ filt_ta['총자산'][year_date]
    filt_ta2 = filt_ta[filt_ta['시총'][index_date] > 0]
    low20 = int(len(filt_ta2) * 0.2)
    filt_tp = filt_ta2.sort_values(by=('시총', index_date))[:low20]
    filt_tp['영흐점수'] = filt_tp['영업활동현흐'][index_date] > 0
    filt_tp['순익점수'] = filt_tp['순익'][index_date] > 0
    filt_tp['F-score'] = filt_tp[['순익점수', '영흐점수']].sum(axis=1)
    f_score = filt_tp[filt_tp['F-score'] == 2]
    kang_super_quality = f_score.sort_values(by='GPA', ascending=False)
    return kang_super_quality[:num]








