import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', password='1234', charset='utf8')
cursor = conn.cursor()

sql = 'create database if not exists flask_db;'
cursor.execute(sql)

cursor.execute('use flask_db;')


sql = """
create table if not exists member (
 `id` int(11) not null auto_increment,
 `userid` varchar(50) not null,
 `pwd` varchar(200) not null,
 `name` varchar(20) default null,
 `email` varchar(50) default null,
 `regdate` datetime default null,
 primary key(`id`) 
)
"""
cursor.execute(sql)


sql = """
insert into member values 
(1, 'aaa', '1234', '아이유', 'aaa@aaa.co.kr', '2019-06-24'),
(2, 'bbb', '1234', '방탄소년', 'bbb@bbb.co.kr', '2019-06-24')
"""
cursor.execute(sql)


sql = """
create table if not exists point_table (
`point_stu_idx` int(11) default null,
`point_stu_grade` varchar(50) default null,
`point_stu_kor` varchar(50) default null
)
"""
cursor.execute(sql)


sql = """
insert into point_table values
(2, '2', '90'),
(2, '3', '94'),
(1, '1', '89');
"""
cursor.execute(sql)


sql = """
CREATE TABLE IF NOT EXISTS student_table (
  `stu_idx` int(11) NOT NULL AUTO_INCREMENT,
  `stu_name` varchar(50) DEFAULT NULL,
  `stu_age` varchar(3) DEFAULT NULL,
  `stu_addr` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`stu_idx`)
) ;
"""
cursor.execute(sql)


sql = """
INSERT INTO student_table (`stu_idx`, `stu_name`, `stu_age`, `stu_addr`) VALUES
	(1, '홍길동', '18', '서울시'),
	(2, '김개똥', '19', '서울시');
"""
cursor.execute(sql)


conn.commit()
conn.close()