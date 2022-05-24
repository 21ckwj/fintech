from flask import Flask, render_template

import selenium
from bs4 import BeautifulSoup
from selenium import webdriver

import time

app = Flask(__name__)

@app.route('/daum_news')
def daum():
    url = 'https://www.daum.net'

    driver = webdriver.Chrome()

    time.sleep(1)
    driver.get(url)
    time.sleep(3)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    print(soup)

    myList = []

    path = '#news > div.news_prime.news_tab2 > div > ul > li > a'
    for i in soup.select(path):
        myList.append(i.text)
        print(i.text)

    return render_template('index1.html', list=myList)

@app.route('/about')
def about():
    return "about page입니다.."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')