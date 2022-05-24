import pymysql

def get_connection():

    conn = pymysql.connect(host='127.0.0.1', user='root', password='1234', charset='utf8')
    return conn   


def get_student_list(stu_name):
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('use flask_db;')
    
    
    sql = """
    select stu_idx, stu_name, stu_age, stu_addr from student_table
    """
    if stu_name != None and len(stu_name) > 0:
        sql += ' where stu_name like %s'
        
    sql += ' order by stu_idx desc;'
        
 
    if stu_name != None and len(stu_name) > 0:
        cursor.execute(sql, (f'%{stu_name}%'))
#         cursor.execute(sql.format(stu_name))
        
    else:
        cursor.execute(sql)
        
    result = cursor.fetchall()
    
    temp_list = []
    
    for row in result :
        temp_dict = {}
        temp_dict['stu_idx'] = row[0]
        temp_dict['stu_name'] = row[1]
        temp_dict['stu_age'] = row[2]
        temp_dict['stu_addr'] = row[3]
        
        temp_list.append(temp_dict)
        
    conn.close()
    
    return temp_list



def add_student(stu_name, stu_age, stu_addr):
    sql = """
    insert into student_table (stu_name, stu_age, stu_addr) values
    (%s, %s, %s) """
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('use flask_db;')    
    
    cursor.execute(sql, (stu_name, stu_age, stu_addr))
    conn.commit()
    
    sql2 = 'select max(stu_idx) from student_table;'
    cursor.execute(sql2)
    result = cursor.fetchone()
    idx = result[0]
    
    conn.close()
    
    return idx
    

def get_student_info(stu_idx):
    sql = """
    select stu_name, stu_age, stu_addr from student_table where stu_idx=%s
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('use flask_db;')     
    
    cursor.execute(sql, (stu_idx))
    result = cursor.fetchone()
    
    result_dic = {}
    result_dic['stu_idx'] = stu_idx
    result_dic['stu_name'] = result[0]
    result_dic['stu_age'] = result[1]
    result_dic['stu_addr'] = result[2]
    
    conn.close()
    
    return result_dic
    


def add_point(point_stu_idx, point_stu_grade, point_stu_kor):
    
    sql = """
    insert into point_table values (%s, %s, %s)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('use flask_db;')     
    
    cursor.execute(sql, (point_stu_idx, point_stu_grade, point_stu_kor))
    
    conn.commit()
    
    conn.close()
    


def get_point(stu_idx):
    
    sql = """
    select point_stu_grade, point_stu_kor from point_table
    where point_stu_idx = %s order by point_stu_grade;
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('use flask_db;')    
    
    cursor.execute(sql, (stu_idx))
#     cursor.execute(sql.format(stu_idx))
    
    result = cursor.fetchall()
    
    result_list = []
    
    for result_dic in result:
        temp_dic = {}
        temp_dic['point_stu_grade'] = result_dic[0]
        temp_dic['point_stu_kor'] = result_dic[1]
        
        result_list.append(temp_dic)
        
    conn.close()
    
    return result_list































