from flask import Flask, render_template

app = Flask(__name__)

@app.route('/hello_loop')
def hello_loop():
    value_list = ['박민수', '김윤아','한철호']
    return render_template('loop.html', values=value_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')

# ctrl + f5