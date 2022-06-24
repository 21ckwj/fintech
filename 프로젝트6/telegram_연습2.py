from hashlib import new
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
rct_lst = ['최신뉴스','최근뉴스']
pos_lst = ['긍정뉴스', '호재', '좋은소식','호재는']
neg_lst = ['부정뉴스', '악재','악재는?']
k_lst = ['키워드','관련있어','관련있어?']
news_lst_total = ['최신뉴스','최근뉴스','긍정뉴스', '호재', '좋은소식','호재는','부정뉴스', '악재','악재는?','키워드','관련있어','관련있어?']


reco_lst1 = ['살만한 주식 뭐 있어?','살만한 종목 뭐 있어?']
reco_lst2 = ['어떤 업종이 괜찮아?','괜찮은 업종이 어떨꺼 같아?','오르는 업종이 어떤거야?']
reco_lst3 = ['코스피가 어떻게 될꺼 같아?','코스피 앞으로 어떻게 될까?','주식시장 앞으로 어떻게 될것 같아?']

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
    for i in range(len(df_senti[:num])):
        title = df_senti['뉴스제목'].iloc[i]
        news_url = df_senti['url'].iloc[i]
        output_result += title + "\n" + news_url + "\n\n"
        
    return df_senti[:num], output_result

# 토크나이져 감정분류모델 불러오기
senti_model, senti_tokenizer = call_senti_model()

# 뉴스 총합 함수
def news_info(corp,word,rct_lst,pos_lst,neg_lst,k_lst,id = 5322933876):
    
    # 최근뉴스
    if word in rct_lst:
        _, recent_news = crawl_news(corp)
        bot.send_message(chat_id=id, text= recent_news)

    # 긍정뉴스 호재있어??
    elif word in pos_lst:
        _,pos_news = bert_clsfy_news(corp,senti_model=senti_model,tokenizer=senti_tokenizer,senti=1,page=5)
        bot.send_message(chat_id=id, text= pos_news)

    # 부정뉴스
    elif word in neg_lst:
        _,neg_text = bert_clsfy_news(corp,senti_model=senti_model,tokenizer=senti_tokenizer,senti=2,page=5)
        if neg_text =='':
            bot.send_message(chat_id=id, text=f'최근 {corp} 는 악재가 없습니다')
        else:
            bot.send_message(chat_id=id, text=neg_text)

    # 삼성전자 뭐랑관련있어? 최근 이슈? 
    elif word in k_lst:
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



####################################

 
######## 텔레그램 관련 코드 ########
token = "5403110188:AAEbcgi6cDNmgdRHERhGhprFQgMUHzi-rtI"
id = 5322933876
 
bot = telegram.Bot(token)
bot1 = TelegramBot(token,id)
info_message = '''안녕하세요 금융챗봇 고슴도치입니다~'''
bot.sendMessage(chat_id=id, text=info_message)
 
updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher
updater.start_polling()
 
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
        if user_text in reco_lst1:
            bot.send_message(chat_id=id, text='고슴도치가 추천해드리겠습니다') # 답장 보내기

        # 업종이 어떤거야?
        elif user_text in reco_lst2:
            bot.send_message(chat_id=id, text='섬유업종 지수상승이 예상됩니다') # 답장 보내기

        # 코스피 지수 or 주식시장
        elif user_text in reco_lst3:
            bot.send_message(chat_id=id, text='코스피지수가 상승할 예정입니다') # 답장 보내기

        

        # gpt일반대화
        else:
            a = finance_gpt(user_text,tokenizer=koGPT2_TOKENIZER,model=model1)
            bot.send_message(chat_id=id, text=a) # 답장 보내기
            
    
    # 종목명 있으면
    else:
        for word in user_words:
            if word in news_lst_total:
                news_info(corp,word,rct_lst,pos_lst,neg_lst,k_lst)
                break
            

        else:
            bot.send_message(chat_id=id, text = '죄송해요 다시 한번 물어봐주세요')
                

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)
####################################



