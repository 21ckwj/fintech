from flask import Flask, render_template, request
import database as db


app = Flask(__name__)

@app.route('/')
def index() :
    html = render_template('index.html')
    return html

@app.route('/student_list')
def student_list() :
    stu_name = request.args.get('stu_name')
    stu_list = db.get_student_list(stu_name)

    html = render_template('student_list.html', data_list=stu_list)
    return html

@app.route('/show_point')
def show_point() :
    html = render_template('show_point.html')
    return html

@app.route('/student_info', methods=['get', 'post'])
def student_info() :
    stu_idx = request.args.get('stu_idx')
    result_dic = db.get_student_info(stu_idx)
    result_list = db.get_point(stu_idx)

    html = render_template('student_info.html', data_dic=result_dic, data_list=result_list)
    return html

@app.route('/add_student')
def add_student() :
    html = render_template('add_student.html')
    return html

@app.route('/add_point')
def add_point() :
    stu_idx = request.args.get('stu_idx')
    temp_dic = {}
    temp_dic['stu_idx'] = stu_idx

    html = render_template('add_point.html', data_dic=temp_dic)
    return html

@app.route('/add_student_pro', methods=['post'])
def add_student_pro() :
    stu_name = request.form['stu_name']
    stu_age = request.form['stu_age']
    stu_addr = request.form['stu_addr']


    idx = db.add_student(stu_name, stu_age, stu_addr)

    result_dic = { 'stu_idx' : idx}

    html = render_template('add_student_pro.html', data_dic=result_dic)
    return html

@app.route('/add_point_pro', methods=['post'])
def add_point_pro() :
    point_stu_grade = request.form['point_stu_grade']
    point_stu_kor = request.form['point_stu_kor']
    point_stu_idx = request.form['point_stu_idx']
    db.add_point(point_stu_idx, point_stu_grade, point_stu_kor)

    temp_dic = {}
    temp_dic['stu_idx'] = point_stu_idx
    html = render_template('add_point_pro.html', data_dic=temp_dic)
    return html

app.run(host='0.0.0.0')


#




