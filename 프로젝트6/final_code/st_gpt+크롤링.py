import math
import numpy as np
import pandas as pd
import random
import re
import torch
import urllib.request
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import os 

import streamlit as st
from streamlit_chat import message as st_message

kospi_list = pd.read_csv('./data/recent_kospi_list.csv')
corp_list = kospi_list['Name']
content_lst = ['최신뉴스','최근뉴스']

#크롤링 함수

def crawl_news(corp,page=1,bgn_date='2022.03.01',end_date='2022.03.30'):
    
    bgn_date1 = bgn_date
    bgn_date2 = bgn_date.replace('.','')
    end_date1 = end_date
    end_date2 = end_date.replace('.','')
    
    title_lst = []
    summary_lst = []
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

            #요약
            summary = li.select('div.news_dsc > div > a')[0].text
            summary_lst.append(summary)

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
    
    return df

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

if 'past' not in st.session_state: # 내 입력채팅값 저장할 리스트
        st.session_state['past'] = [] 

if 'generated' not in st.session_state: # 챗봇채팅값 저장할 리스트
    st.session_state['generated'] = []

st.title("금융챗봇 고슴도치")

placeholder = st.empty() # 채팅 입력창을 아래위치로 내려주기위해 빈 부분을 하나 만듬

with st.form('form', clear_on_submit=True): # 채팅 입력창 생성
        user_input = st.text_input('당신: ', '') # 입력부분
        submitted = st.form_submit_button('전송') # 전송 버튼

if submitted and user_input:
    user_input1 = user_input.strip() # 채팅 입력값 및 여백제거
    user_words = user_input1.split()  # 입력값 단어로 쪼개기

    for word in user_words:
        # 종목명을 포함한다면
        if word in corp_list.tolist():
            corp = [w for w in user_words if w in corp_list.tolist()]
            user_words.remove(corp[0])
            break
        
        # 종목명을 포함하지 않는다면
        else:
            a = ""
            while 1:
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + user_input1 + SENT + A_TKN + a)).unsqueeze(dim=0)
                pred = model1(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")

            chatbot_output1 = a.strip() # text generation된 값 및 여백 제거

            st.session_state.past.append(user_input1) # 입력값을 past 에 append -> 채팅 로그값 저장을 위해
            st.session_state.generated.append(chatbot_output1)
            break


    # 종목명을 포함할때
    for word1 in user_words:
        # 최신뉴스,최근뉴스를 포함한다면
        if word1 in content_lst:
            df = crawl_news(corp)
            news_title = df['뉴스제목'].tolist()
            chatbot_output1 = '//'.join(news_title)
            st.dataframe(df)
            st.session_state.past.append(user_input1)
            st.session_state.generated.append(chatbot_output1)
            break
   
        
        
with placeholder.container(): # 리스트에 append된 채팅입력과 로봇출력을 리스트에서 꺼내서 메세지로 출력
    for i in range(len(st.session_state['past'])):
        st_message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i:
            st_message(st.session_state['generated'][i], key=str(i) + '_bot')

