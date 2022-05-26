from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

app = Flask(__name__)


#엑셀 import
# from openpyxl import Workbook
# write_wb = Workbook()
# write_ws = write_wb.active

#셀레니움
from selenium import webdriver

@app.route('/')
def hello_world():
    return render_template("index_crawling.html")


@app.route('/result', methods=['POST'])
def result() :
    
    if request.method == 'POST' :

        keyword = request.form['input1']
        page = request.form['input2']

        daum_list = []

        driver = webdriver.Chrome('/chromedriver')

        for i in range(1, int(page)+1) :
            url = 'https://search.daum.net/search?w=news&DA=PGD&enc=utf8&cluster=y&cluster_page=1&q={}&p={}'.format(keyword, i)
            driver.get(url)
            time.sleep(2)

            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
           
            for i in soup.select('#newsColl > div.cont_divider > ul > li > div.wrap_cont > a') :
                daum_list.append(i.text)

        return render_template('daum_crawling.html', daum_list = daum_list)


@app.route('/naver_shopping', methods=['POST'])
def naver() :

    search = request.form['input3']

    total_list = []
    total_desc = []

    driver = webdriver.Chrome('/chromedriver')

    driver.implicitly_wait(3)

    driver.get("https://search.shopping.naver.com/search/all_search.nhn?query=" + search + "&cat_id=&frm=NVSHATC")
    driver.implicitly_wait(3)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # for i in soup.select("#_search_list > div.search_list.basis > ul > li") :
    #     print(i.find("a", class_="link").text)

    for i in soup.select("#__next > div > div.style_container__1YjHN > div > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > li > div > div.basicList_info_area__17Xyo > div.basicList_title__3P9Q7 > a") :
            print(i.text)
            total_list.append(i.text)

    path = '#__next > div > div.style_container__1YjHN > div.style_inner__18zZX > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > li > div > div.basicList_info_area__17Xyo > div.basicList_desc__2-tko > div.basicList_detail_box__3ta3h'
    for i in soup.select(path):
        print(i.text)
        total_desc.append(i.text)

    print()
    print("---백화점/홈쇼핑--")
    print()

    driver.find_element_by_css_selector("#__next > div > div.style_container__1YjHN > div > div.style_content_wrap__1PzEo > div.style_content__2T20F > div.seller_filter_area > ul > li:nth-child(4) > a").click()
    driver.implicitly_wait(3)
    time.sleep(2)

    department_list = []
    department_desc = []

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    for i in soup.select("#__next > div > div.style_container__1YjHN > div > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > li > div > div.basicList_info_area__17Xyo > div.basicList_title__3P9Q7 > a") :
            print(i.text)
            department_list.append(i.text)

    path = '#__next > div > div.style_container__1YjHN > div.style_inner__18zZX > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > li > div > div.basicList_info_area__17Xyo > div.basicList_desc__2-tko.basicList_max__boWiv > div.basicList_detail_box__3ta3h'
    for i in soup.select(path):
        print(i.text)
        department_desc.append(i.text)

    time.sleep(1)

    print("---overseas---")

    driver.find_element_by_css_selector("#__next > div > div.style_container__1YjHN > div.style_inner__18zZX > div.style_content_wrap__1PzEo > div.style_content__2T20F > div.seller_filter_area > ul > li:nth-child(6) > a").click()
    driver.implicitly_wait(3)
    time.sleep(2)

    oversea_list = []
    oversea_desc = []

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    for i in soup.select("#__next > div > div.style_container__1YjHN > div.style_inner__18zZX > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > li > div > div.basicList_info_area__17Xyo > div.basicList_title__3P9Q7 > a") :
            print(i.text)
            oversea_list.append(i.text)

    path = '#__next > div > div.style_container__1YjHN > div.style_inner__18zZX > div.style_content_wrap__1PzEo > div.style_content__2T20F > ul > div > div > li > div > div.basicList_info_area__17Xyo > div.basicList_desc__2-tko > div.basicList_detail_box__3ta3h'
    for i in soup.select(path):
        print(i.text)
        oversea_desc.append(i.text)

    driver.close()

    return render_template('naver_shopping.html', \
        total_list=total_list, total_desc=total_desc, total_len=len(total_list), \
        department_list=department_list, department_desc=department_desc, department_len=len(department_list), \
        oversea_list=oversea_list, oversea_desc=oversea_desc, oversea_len=len(oversea_list))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8080')
