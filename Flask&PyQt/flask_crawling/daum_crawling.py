from flask import Flask, render_template, request

import selenium
from bs4 import BeautifulSoup
from selenium import webdriver

import time
import requests

app = Flask(__name__)



@app.route('/')
def hello_world():
    return render_template("index6.html")


@app.route('/result', methods=['POST'])
def result() :

    if request.method == 'POST' :

        keyword = request.form['input1']
        page = request.form['input2']

        daum_list = []

        driver = webdriver.Chrome()

        for i in range(1, int(page)+1) :
            url = 'https://search.daum.net/search?w=news&DA=PGD&enc=utf8&cluster=y&cluster_page=1&q={}&p={}'.format(keyword, i)
            driver.get(url)
            time.sleep(2)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
           
            for i in soup.select('#newsColl > div.cont_divider > ul > li > div.wrap_cont > a') :
                daum_list.append(i.text)

        return render_template('result1.html', daum_list = daum_list)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8080')
