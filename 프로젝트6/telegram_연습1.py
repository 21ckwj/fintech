import math
import numpy as np
import pandas as pd
import re
import os

# telegram
import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters
from telegram import chat

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

from bob_telegram_tools.bot import TelegramBot
import matplotlib.pyplot as plt



# Q_TKN = "<usr>"
# A_TKN = "<sys>"
# BOS = '</s>'
# EOS = '</s>'
# MASK = '<unused0>'
# SENT = '<unused1>'
# PAD = '<pad>'

# model1 = torch.load('./data/GPT2_model/gpt_finance.pth')
# koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
#             bos_token=BOS, eos_token=EOS, unk_token='<unk>',
#             pad_token=PAD, mask_token=MASK) 

kospi_list = pd.read_csv('./data/recent_kospi_list.csv')
corp_list = kospi_list['Name']
content_lst = ['최신뉴스','최근뉴스']

reco_lst1 = ['살만한 주식 뭐 있어?','살만한 종목 뭐 있어?']


 ######## 크롤링 관련 함수 ########

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
        
    return output_result

# def finance_gpt(user_text,tokenizer, model):
    
#     with torch.no_grad():
#         answer = ""
#         while 1:
#             input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + user_text + SENT + A_TKN + answer)).unsqueeze(dim=0)
#             pred = model(input_ids)
#             pred = pred.logits
#             gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
#             if gen == EOS:
#                 break
#             answer += gen.replace("▁", " ")
#         answer = answer.strip()
#     return answer
####################################
 
from konlpy.tag import Hannanum
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
df_n = crawl_news('삼성전자',10)
df_n

def kw_crawl_news(corp,bgn_date='2022-03-01',end_date='2022-03-30'):

    df_n = crawl_news(corp,10)
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

kw_crawl_news('삼성전자',10)


######## 텔레그램 관련 코드 ########
token = "5403110188:AAEbcgi6cDNmgdRHERhGhprFQgMUHzi-rtI"
id = 5322933876
 
bot = telegram.Bot(token)
bot1 = TelegramBot(token,id)
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')

bot1.send_plot(plt)

info_message = '''안녕하세요 금융챗봇 고슴도치입니다'''
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

        # gpt일반대화
        else:
            a = finance_gpt(user_text,tokenizer=koGPT2_TOKENIZER,model=model1)
            bot.send_message(chat_id=id, text=a) # 답장 보내기
            
    
    # 종목명 있으면
    else:
        for word in user_words:
            # 최신뉴스 최근뉴스 포함시
            if word in content_lst:
                recent_news = crawl_news(corp)
                bot.send_message(chat_id=id, text= recent_news)
                break
            

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)
####################################




