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

import streamlit as st
from streamlit_chat import message as st_message

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
    if user_input1 == 'quit':
        st.stop()

    # output값
    a = ""
    while 1:
        # 일반대화
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
    
with placeholder.container(): # 리스트에 append된 채팅입력과 로봇출력을 리스트에서 꺼내서 메세지로 출력
    for i in range(len(st.session_state['past'])):
        st_message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        if len(st.session_state['generated']) > i:
            st_message(st.session_state['generated'][i], key=str(i) + '_bot')

