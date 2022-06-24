import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pymysql
import MySQLdb

import warnings
warnings.filterwarnings('ignore')

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

DB_info('삼성전자','종가','2022-03-30')
DB_info('삼성전자','종가','2022-03-29')
DB_info('삼성전자','시가','2022-03-30')
DB_info('삼성전자','PER','2022-03-30')
DB_info('삼성전자','영업이익','2022-03-30')
