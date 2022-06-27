import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import matplotlib.colors as mcolors
from matplotlib import cm

# telegram
import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters
from telegram import chat
from bob_telegram_tools.bot import TelegramBot

# 크롤링
from bs4 import BeautifulSoup
import requests
from selenium import webdriver

# gpt함수
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel

# 자연어처리
from konlpy.tag import Hannanum

# BERT 
import tensorflow_addons as tfa
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, TFBertForSequenceClassification

# DB
import pymysql
import MySQLdb

import warnings
warnings.filterwarnings('ignore')


Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

model1 = torch.load('./data/GPT2_model/gpt_finance.pth')
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
    

kospi_list = pd.read_csv('./data/recent_kospi_list.csv')
corp_list = kospi_list['Name']
news_lst1 = ['최신뉴스','최근뉴스']
news_lst2 = ['긍정뉴스', '호재', '좋은소식','호재는']
news_lst3 = ['부정뉴스', '악재','악재는?']
news_lst4 = ['키워드','관련있어','관련있어?']


reco_lst1 = ['살만한', '주식', '뭐있어','종목', '뭐','있어', '뭐야','투자', '종목추천']
reco_lst2 = ['어떤', '업종이', '괜찮아', '업종','업종추천']
reco_lst3 = ['코스피','주식시장']
reco_lst4 = ['시가','종가','영업이익','PER']
reco_lst5 = ['어떨꺼', '같아','어때','보여','오를까', '내릴까']

 ######## 함수 ########

def crawl_news(corp,page=1,num=5,bgn_date='2022.03.01',end_date='2022.03.30'):
    
    bgn_date1 = bgn_date
    bgn_date2 = bgn_date.replace('.','')
    end_date1 = end_date
    end_date2 = end_date.replace('.','')
    
    title_lst = []
    url_lst = []
    date_lst = []

    for pg in range(1,page+1):

        page_num = pg *10 - 9

        url = f'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={corp}&sort=0&photo=0&field=0&pd=3&ds={bgn_date1}&de={end_date1}&cluster_rank=24&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{bgn_date2}to{end_date2},a:all&start={page_num}'
        res = requests.get(url)
        soup = BeautifulSoup(res.text , 'html.parser')
        lis = soup.select('#main_pack > section > div > div.group_news > ul>li')

        for li in lis:
            #제목
            title = li.select('div.news_wrap.api_ani_send > div > a')[0].text

            title_lst.append(title)

            # url
            url_path = li.select('div.news_wrap.api_ani_send > div > a')[0]['href']
            url_lst.append(url_path)

            #날짜

            if len(li.select('div.news_info > div.info_group > span'))==1:
                date = li.select('div.news_info > div.info_group > span')[0].text
                date_lst.append(date)


            if len(li.select('div.news_info > div.info_group > span'))==2:
                date = li.select('div.news_info > div.info_group > span')[1].text
                date_lst.append(date)
    
    df = pd.DataFrame({'날짜':date_lst,'뉴스제목':title_lst,'url':url_lst})
    
    output_result = ''
    for i in range(len(df)):
        title = df['뉴스제목'].iloc[i]
        news_url = df['url'].iloc[i]
        output_result += title + "\n" + news_url + "\n\n"
        if i == num:
            break
        
    return df, output_result

def finance_gpt(user_text,tokenizer, model):
    
    with torch.no_grad():
        answer = ""
        while 1:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + user_text + SENT + A_TKN + answer)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            answer += gen.replace("▁", " ")
        answer = answer.strip()
    return answer

# 한글,영어만 남기기
def clean_text(docs):
    docs = re.sub('[^가-힣A-Za-z ]', '', str(docs))
    docs = re.sub('\s+', ' ', docs)
    docs = '' if docs== ' ' else docs
    return docs

# 명사 추출
def han_noun(docs):
    han = Hannanum()
    docs = han.nouns(docs)
    return docs

# 불용어 제거+ 한글자 이상만 남기기
stw_list = pd.read_csv('./data/stopwords-ko.txt')
def remove_stwords(docs):
    docs = [w for w in docs if not w in stw_list]
    docs = '' if docs== ' ' else docs
    docs = [w for w in docs if len(w)>1]
    return docs

# 크롤링 뉴스 키워드 선출
def kw_crawl_news(corp,bgn_date='2022-03-01',end_date='2022-03-30'):
    df_n, _ = crawl_news(corp,page=10)
    df_n = df_n.set_index('날짜')
    df_n.index = pd.to_datetime(df_n.index)
    df_n['뉴스'] = df_n['뉴스제목'].apply(clean_text)
    df_n['뉴스'] = df_n['뉴스'].apply(han_noun)
    df_n['뉴스'] =  df_n['뉴스'].apply(remove_stwords)
    df_stw = pd.read_csv('./data/뉴스불용어2.csv',index_col=1)
    stw_lst = df_stw['불용어'].tolist()
    kw_dict = dict()

    token_lst = df_n['뉴스'].loc[bgn_date:end_date]
            
    for tokens in token_lst:
        for word in tokens:
                
            if not word in kw_dict.keys():
                kw_dict[word] = 1
            else:
                kw_dict[word] += 1

    kw_dict = dict(sorted(kw_dict.items(), key = lambda x: x[1],reverse=True))
    keys = pd.Series(kw_dict.keys()).tolist()
    for key in keys:
        if (key in stw_lst) | (key == corp):
            del kw_dict[key]
            
    return kw_dict

# 키워드 시각화
def show_kw(kw_dict,num=10):
    plt.figure(figsize=(10,8))
    plt.rc('ytick', labelsize=15)  # y축 눈금 폰트 크기
    plt.title('키워드 상위 15')
    keyword_list = list(kw_dict.keys())[:num]
    keyword_list.reverse()
    keyword_count = list(kw_dict.values())[:num]
    keyword_count.reverse()
    sns.barplot(x=keyword_count, y=keyword_list)
    plt.gca().invert_yaxis()
    return plt

# 최고 성능의 모델 불러오기
def call_senti_model():
    BEST_MODEL_NAME = './data/model/best_model.h5'
    sentiment_model_best = tf.keras.models.load_model(BEST_MODEL_NAME,
                                                      custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification})
    MODEL_NAME = "klue/bert-base"
    model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, from_pt=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    return sentiment_model_best, tokenizer

# 데이터형태 맞춰주기
def senti_text_data(X_data,tokenizer,MAX_SEQ_LEN=64):
    # BERT 입력으로 들어가는 token, mask, segment, target 저장용 리스트
    tokens, masks, segments, targets = [], [], [], []
    
    for X in tqdm(X_data):
        # token: 입력 문장 토큰화
        token = tokenizer.encode(X, truncation = True, padding = 'max_length', max_length = MAX_SEQ_LEN)
        
        # Mask: 토큰화한 문장 내 패딩이 아닌 경우 1, 패딩인 경우 0으로 초기화
        num_zeros = token.count(0)
        mask = [1] * (MAX_SEQ_LEN - num_zeros) + [0] * num_zeros
        
        # segment: 문장 전후관계 구분: 오직 한 문장이므로 모두 0으로 초기화
        segment = [0]*MAX_SEQ_LEN

        tokens.append(token)
        masks.append(mask)
        segments.append(segment)


    # numpy array로 저장
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

# BERT 긍부정 판별
def bert_clsfy_news(corp,senti_model,tokenizer,senti=1,page=5,num=5): #1 2 0 긍정 부정 중립
    df_n,_ = crawl_news(corp,page)
    df_n = df_n.set_index('날짜')
    df_n.index = pd.to_datetime(df_n.index)
    news_data = df_n['뉴스제목']
    news_x = senti_text_data(news_data,tokenizer)
    predicted_value = senti_model.predict(news_x)
    predicted_label = np.argmax(predicted_value, axis = 1)
    idx = np.where(predicted_label== senti)[::-1]
    df_senti = df_n.iloc[idx][:'2022-03-30'].sort_index(ascending=False)

    output_result = ''
    for i in range(len(df_senti)):
        title = df_senti['뉴스제목'].iloc[i]
        news_url = df_senti['url'].iloc[i]
        output_result += title + "\n" + news_url + "\n\n"
        
    return df_senti[:num], output_result

# 토크나이져 감정분류모델 불러오기
senti_model, senti_tokenizer = call_senti_model()

# DB에서 정보추출
def DB_info(name,db_type,date): #db_type: 종가,시가,PER,영업이익
    end_point = "chatbot-db.c9x08hbiunuu.ap-northeast-2.rds.amazonaws.com"
    port =3306
    user_name = 'root'
    pw ='123123123'

    conn = pymysql.connect(
        host = end_point,
        user = user_name,
        password = pw,
    #     db = db,
        charset='utf8'

    )
    cursor = conn.cursor()

    sql = 'use Chatbot_DB;'
    cursor.execute(sql)

    sql = 'select * from stock_table;'

    stock_table = pd.read_sql(sql, conn)
    stock_df = stock_table[stock_table.name == name]
    stock_df['날짜'] = stock_df['날짜'].astype('datetime64')

    val = str(stock_df[stock_df['날짜'] == date][db_type].values).strip('[]').strip("''")

    if db_type == '영업이익':
        print(f'{name}의 {db_type}은 {val}원 입니다')
    
    elif db_type == 'PER':
        print(f'{name}의 {db_type}는 {val}배 입니다')

    else: # 시가,종가
        print(f'{name}의 {date} {db_type}는 {val}원 입니다')
        
# 종목 추세         
def companyy_predict(name, date):
    
    end_point = "chatbot-db.c9x08hbiunuu.ap-northeast-2.rds.amazonaws.com"
    port =3306
    user_name = 'root'
    pw ='123123123'
    
    conn = pymysql.connect(
    host = end_point,
    user = user_name,
    password = pw,
#     db = db,
    charset='utf8'
    
    )
    cursor = conn.cursor()
    
    sql = 'use Name_Code_DB;'
    cursor.execute(sql)
    
    sql = 'select * from name_table;'

    name_df = pd.read_sql(sql, conn)
    
    for na in range(len(name_df)):
        if name == name_df.iloc[na,1]:
            code = name_df.iloc[na,0]
    
    autoencoder = load_model(f'c:/Users/bitcamp/Desktop/final_data/autoencoder/Conv_2/auto_conv_{code}.h5')

    company_df = fdr.DataReader(code)
    stock_close = company_df[['Close']]

    stock_close.columns = ['price']
    stock_close['pct_change'] = stock_close.price.pct_change()
    stock_close['log_ret'] = np.log(com_close.price) - np.log(stock_close.price.shift(1))
    stock_close.dropna(inplace=True)

    x_test = stock_close[stock_close.index == date]

    window_length = 10
    scaler = MinMaxScaler()
    x_test_nonscaled = np.array([stock_close['log_ret'].values[i-window_length:i].reshape(-1, 1) for i in tqdm(range(window_length+1,len(stock_close['log_ret'])))])
    x_test = np.array([scaler.fit_transform(stock_close['log_ret'].values[i-window_length:i].reshape(-1, 1)) for i in tqdm(range(window_length+1,len(stock_close['log_ret'])))])

    y_predict = autoencoder.predict(x_test)
    labels = [1,0]

    label = labels[y_predict[0].argmax()]
    confidence = y_predict[0][y_predict[0].argmax()]
    # print('{} {:.2f}%'.format(label, confidence * 100))

    if label == 1:
        print(f'{name}는 상승할 예정입니다.')
    else:
        print(f'{name}는 하락할 예정입니다.')
        
# 재무데이터 AI 추천
def company_recomend(date):
    end_point = "chatbot-db.c9x08hbiunuu.ap-northeast-2.rds.amazonaws.com"
    port =3306
    user_name = 'root'
    pw ='123123123'

    conn = pymysql.connect(
        host = end_point,
        user = user_name,
        password = pw,
    #     db = db,
        charset='utf8'

    )
    cursor = conn.cursor()

    sql = 'use Chatbot_DB;'
    cursor.execute(sql)

    sql = 'select * from stock_table;'

    stock_table = pd.read_sql(sql, conn)
    
    
    stock_table['날짜'] = stock_table['날짜'].astype('datetime64')
    stock_df = stock_table[stock_table['날짜'] == date]
    stock_df = stock_df.drop(['BPS', 'PER', 'PBR', 'EPS', 'DIV_', 'DPS', 'quarter','날짜','등락률','name'], axis=1)
    stock_df = stock_df.astype(str).apply(lambda col : col.apply(lambda x: np.nan if x == 'nan' else x))
    stock_df.dropna(inplace=True)

    hoho = defaultdict(list)
    plus_df = pd.DataFrame()
    hoit_df = pd.DataFrame()
    fail_li = []
    code_li = []
    month_li = ['30일','3개월','6개월']
    per_li = ['5per','10per','15per']

    for code in stock_df['code'][:]:
        try:

            df = pd.read_csv(f'c:/Users/bitcamp/Desktop/final_data/머신러닝_재무데이터/머신러닝_train_test/{month}/test/{code}.csv', index_col=0)
            df.drop(['date','등락률','종목명','수익률','quarter','5per','10per','15per'], axis=1, inplace=True)
            df.rename(columns={'법인세차감전 순이익':'법인세차감전_순이익'}, inplace=True)

            ya_df = pd.concat([stock_df[['code']],stock_df[list(df.columns)]], axis=1)
            x_test = ya_df[ya_df.code == code].iloc[0,1:]

            for monmon in month_li[:]:
                for per in per_li[:]:

                        with open(f'c:/Users/bitcamp/Desktop/final_data/머신러닝_모델/{monmon}_{per}/Randomforest/{code}.pkl','rb') as f:
                                rfc = pickle.load(f)

                        pred = rfc.predict([x_test])

                        col = monmon+'_'+per

                        hoho[col].append(pred[0])
            code_li.append(code)

        except:
#             print(code)
            pass

    plus_df = pd.DataFrame({'code':code_li})

    for col,val in hoho.items():
        plus_df[col] = val
    
#     print(plus_df.head())
    
    max_plus_df = plus_df[plus_df['30일_15per']== 1]
    max_plus_df.reset_index(drop=True, inplace=True)

    pre_df = pd.read_csv('c:/Users/bitcamp/Desktop/final_data/모델정리/종목별_머신러닝/시연용/rfc_30_15.csv', index_col=0)
    pre_df.code = pre_df.code.apply(lambda x:f'{x:06}')

    company_li = []
    for pre in range(len(pre_df)):
        for coco in range(len(max_plus_df)):
            if max_plus_df.iloc[coco, 0] == pre_df.iloc[pre, 0]:
                company_li.append(pre_df.iloc[pre,1])

    return company_li[1].strip('[]').strip("''"), company_li[2].strip('[]').strip("''"), company_li[3].strip('[]').strip("''")

# 코스피 업종 추천
def kospi_kind_recomend():
    kospi_df = pd.read_csv('c:/Users/bitcamp/Desktop/final_data/업종별/월봉_21_8.csv')

    kospi_corr_df = kospi_df.corr()
    kospi_corr_df = kospi_corr_df[['코스피_5']]

    kospi_corr_df.reset_index(drop=False, inplace=True)
    kospi_corr_df.columns = ['업종', 'kospi']

    # kospi_corr_df.sort_values(by=['kospi'], ascending=False)
    kospi_kind_df = kospi_corr_df.sort_values(by=['kospi'])
    kind_recomend = kospi_kind_df.iloc[0,0]
    return kind_recomend.split('_')[0]

# 코스피 추세 예측
def kospi_predict(date):
    kospi = fdr.DataReader('KS11')
    kospi_df = kospi[['Close']]

    stock = pd.DataFrame()
    stock['y'] = kospi_df[['Close']]
    stock['ds'] = kospi_df.index

    stock.reset_index(drop=True, inplace=True)

    train = stock[stock.ds < '2021-01']
    test = stock[stock.ds > '2021-01']

    df_prophet = Prophet(changepoint_prior_scale=0.15, daily_seasonality=True)
    df_prophet.fit(train)

    fcast_time = 100
    df_forecast = df_prophet.make_future_dataframe(periods=fcast_time, freq='D')
    # df_forecast.tail(50)
    df_forecast = df_prophet.predict(df_forecast)

    kospi_updow_df = df_forecast[['ds','yhat','yhat_lower','yhat_upper']]
#     kospi_updow_df['ds'] = kospi_updow_df['ds'].astype('datetime64')

    kospi_1 = str(kospi_updow_df[kospi_updow_df['ds'] == date]['yhat_upper'].values)
    kospi_1 = kospi_1.strip('[]')[:4]

    kospi_2 = str(kospi_updow_df[kospi_updow_df['ds'] == date]['yhat_upper'].values)
    kospi_2 = kospi_2.strip('[]')[:4]

    n = int(kospi_2) - int(kospi_1)

    if n > 0:
        print('코스피지수가 상승할 예정입니다.')
    else :
        print('코스피지수가 하락할 예정입니다.')

# 볼린져밴드 전략
def bb(date):
    import os
    import pandas as pd
    import numpy as np
    import FinanceDataReader as fdr 
    df_krx = fdr.StockListing('KOSPI')
    file_nam = os.listdir('./data/재무+주가_511_last/재무+주가_511_last/')
#     file_nam[0].split('.')[0]
    bb_lis = []
    name_lis = []
    for i in file_nam:
        try:
            df = pd.read_csv('./data/재무+주가_511_last/재무+주가_511_last/'+i,index_col=0)
            df = df.set_index('date')
            
            price_df = df[['수정종가','종목명']] 
#             bb = bollinger_band(price_df,20,2)
            bb = price_df.copy()
            bb['center'] = price_df['수정종가'].rolling(20).mean() #중앙 이동평균선
            bb['ub'] = bb['center'] + 2 * price_df['수정종가'].rolling(20).std() # 상단 밴드
            bb['lb'] = bb['center'] - 2 * price_df['수정종가'].rolling(20).std() # 하단 밴드

            book = bb[['수정종가','종목명']].copy()
            book['trade'] = ''

            for i in book.index:
                if bb.loc[i, 'lb'] > bb.loc[i, '수정종가']:
                    book.loc[i, 'trade'] = 'buy'
            bb_lis.append(book[book['trade'] == 'buy'])
        except:
            pass
    trade_df = pd.concat(bb_lis)
    trade_df.index = pd.to_datetime(trade_df.index)
    ticker = trade_df.loc[date,'종목명'].apply(lambda x:str(x).zfill(6)).to_list()
    for i in ticker:
        names = df_krx[df_krx['Symbol'] == i]['Name'].to_list()
        name_lis.append(names)
    return name_lis

# 마법의 공식
def magic(date):
    
    import FinanceDataReader as fdr 
    import os 
    import pandas as pd
    import numpy as np
    
    df_krx = fdr.StockListing('KOSPI')
    name_lis1=[]

    file_lis = os.listdir('./data/재무+주가_511_last/재무+주가_511_last/')
    lis = []
    for i in file_lis:
        df = pd.read_csv('./data/재무+주가_511_last/재무+주가_511_last/'+i,index_col=0)
        lis.append(df)

    df = pd.concat(lis)
    df = df.set_index('date')

    df_2022 = df.loc[date]

    df_2022['ROA'] = pd.to_numeric(df_2022['당기순이익'],errors='coerce') /pd.to_numeric(df_2022['자산총계'],errors='coerce') *100

    per = pd.to_numeric(df_2022['PER'])
    roa = pd.to_numeric(df_2022['ROA'])

#     per_rank = sort_value(per, asc=True, standard=0 ) # PER 지표값을 기준으로 순위 정렬 및 0 미만 값 제거
    
    s_value_mask = per.mask(per < 0, np.nan)
    s_value_mask_rank = s_value_mask.rank(ascending=True, na_option='bottom')
    per_rank = s_value_mask_rank
    
    s_value_mask1 = roa.mask(roa < 0, np.nan)
    s_value_mask_rank1 = s_value_mask1.rank(ascending=False, na_option='bottom')
    roa_rank = s_value_mask_rank1
    
#     roa_rank = sort_value(roa, asc=False, standard=0 )# ROA 지표값을 기준으로 순위 정렬 및 0 미만 값 제거
     
    result_rank = per_rank + roa_rank   # PER 순위 ROA 순위 합산
#     result_rank = sort_value(result_rank, asc=True)  # 합산 순위 정렬
    
    a = result_rank.mask(result_rank < 0, np.nan)
    s_value_mask_rank2 = a.rank(ascending=True, na_option='bottom')
    result_rank = s_value_mask_rank2
    
    result_rank = result_rank.where(result_rank <= 10, 0)  # 합산 순위 필터링
    result_rank = result_rank.mask(result_rank > 0, 1)  # 순위 제거

    mf_df = df_2022.loc[result_rank > 0,['종목명','시가총액']].copy() # 선택된 종목 데이터프레임 복사
    # mf_stock_list = df_2022.loc[result_rank > 0, '종목명'].values # 선택된 종목명 추출
    mf_stock_list = df_2022.loc[result_rank > 0, '종목명'].apply(lambda x: str(x).split('.')[0]).apply(lambda x: x.zfill(6)).to_list() # 선택된 종목명 추출
    # ticker = trade_df.loc[date,'종목명'].apply(lambda x:str(x).zfill(6)).to_list()
    for i in mf_stock_list:
        names = df_krx[df_krx['Symbol'] == i]['Name'].to_list()
        name_lis1.append(names)
        
    return name_lis1

#  파마+LSV 
def famaLSV(date):
    import pandas as pd
    import numpy as np
    import os 
    import FinanceDataReader as fdr 
    name_lis3=[]
    fdr_df = fdr.StockListing('KOSPI')
    file_lis = os.listdir('./data/재무+주가_511_last/재무+주가_511_last/')
    lis = []
    for i in file_lis:
        df = pd.read_csv('./data/재무+주가_511_last/재무+주가_511_last/'+i,index_col=0)
        lis.append(df)

    df = pd.concat(lis)
    df = df.set_index('date')

    df_2016 = df.loc[date]
    df_2016['PSR'] = df_2016['시가총액'] / df_2016['매출액'] 
    # df_2016['시총_rank'] = df_2016.loc['2016-01-04':,'시가총액'].rank()
    # df_2016['pbr_rank'] = df_2016.loc['2016-01-04':,'PBR'].rank()
    # df_2016['psr_rank'] = df_2016.loc['2016-01-04':,'PSR'].rank()
    df_2016['시총_rank'] = df_2016['시가총액'].rank()
    df_2016['pbr_rank'] = df_2016['PBR'].rank()
    df_2016['psr_rank'] = df_2016['PSR'].rank()

    df_2016['total_rank'] = df_2016['pbr_rank'] + df_2016['psr_rank'] + df_2016['시총_rank']
    종목명2016 = df_2016.sort_values('total_rank')[:5]['종목명'].apply(lambda x:str(x).split('.')[0]).apply(lambda x:str(x).zfill(6)).to_list()
#     종목명2016 = df_2016.sort_values('total_rank')[:5]['종목명']
    for i in 종목명2016:
        names = fdr_df[fdr_df['Symbol'] == i]['Name'].to_list()
        name_lis3.append(names)
    
    return name_lis3



####################################

 
######## 텔레그램 관련 코드 ########
token = "5403110188:AAEbcgi6cDNmgdRHERhGhprFQgMUHzi-rtI"
id = 5322933876
 
bot = telegram.Bot(token)
# bot1 = TelegramBot(token,id)
info_message = '''안녕하세요 금융챗봇 고슴도치입니다~'''
bot.sendMessage(chat_id=id, text=info_message)
 
updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher
updater.start_polling()

if '어제' in user_words:
    date = '2022-03-29'
else : 
    date = '2022-03-30'

for com in user_words:
    if com in kospi_list:
        company = com
    else:
        bot.send_message(chat_id=id, text='회사명을 다시 써주세요')
 
 ### 챗봇 답장

def handler(update, context): 
    user_text = update.message.text # 사용자가 보낸 메세지 user_text 변수에 저장
    user_words = user_text.split()
    
    # 종목명 찾기
    for word in user_words:
        # 종목명을 포함한다면
        if word in corp_list.tolist():
            corp = [w for w in user_words if w in corp_list.tolist()][0]
            break
        else:
            corp='종목명 없음'

    # 종목명 없으면 gpt or 추천
    if corp == '종목명 없음':
        # 살만한 주식 뭐있어?
        if user_words in reco_lst1:
            val_text = company_recomend(date)
            bot.send_message(chat_id=id, text=val_text) # 답장 보내기

        # 업종이 어떤거야?
        elif user_words in reco_lst2:
            val_text = kospi_kind_recomend()
            bot.send_message(chat_id=id, text=val_text) # 답장 보내기

        # 코스피 지수 or 주식시장
        elif user_words in reco_lst3:
            val_text = kospi_predict(date)
            bot.send_message(chat_id=id, text=val_text) # 답장 보내기
            

           
            
        
        

        # gpt일반대화
        else:
#             pass
            a = finance_gpt(user_text,tokenizer=koGPT2_TOKENIZER,model=model1)
            bot.send_message(chat_id=id, text=a) # 답장 보내기
            
    
    # 종목명 있으면
    else:
        for word in user_words:
            # 최신뉴스 최근뉴스 포함시
            if word in news_lst1:
                _, recent_news = crawl_news(corp)
                bot.send_message(chat_id=id, text= recent_news)
                break
            # 긍정뉴스 호재있어??
            elif word in news_lst2:
                _,pos_news = bert_clsfy_news(corp,senti_model=senti_model,tokenizer=senti_tokenizer,senti=1,page=5)
                bot.send_message(chat_id=id, text= pos_news)
                break
            # 부정뉴스
            elif word in news_lst3:
                _,neg_text = bert_clsfy_news(corp,senti_model=senti_model,tokenizer=senti_tokenizer,senti=2,page=5)
                bot.send_message(chat_id=id, text= neg_text)
                break
            # 삼성전자 뭐랑관련있어? 최근 이슈? 
            elif word in news_lst4:
                kw_dict = kw_crawl_news(corp=corp)
                keyword_list = list(kw_dict.keys())[:10]
                kw_lst = []
                for i in range(len(keyword_list)):
                    kw_lst.append('#'+keyword_list[i])
                keyword_10 = ', '.join(kw_lst)
                keyword_text = f'{corp}는 최근 {keyword_10} 과 관련이 있습니다'
                plot = show_kw(kw_dict,num=15)
                bot.send_message(chat_id=id, text= keyword_text)
                bot1.send_plot(plot)
                break
        

                
        # 삼성전자 얼마야?
        for word in range(len(user_words)):
            if user_words[word] in reco_lst4:
                val_text = DB_info(company,user_words[word], date)
                bot.send_message(chat_id=id, text=val_text) # 답장 보내기
            
            
        # 삼성전자 어떨꺼 같아?
        if user_words in reco_lst5:
            val_text = company_predict(company,date)
            bot.send_message(chat_id=id, text=val_text) # 답장 보내기
            
            

        


            # else:
            #     bot.send_message(chat_id=id, text = '죄송해요 다시 한번 물어봐주세요')
                

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)
####################################


