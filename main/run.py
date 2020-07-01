import flask

app = flask.Flask(__name__)


@app.route('/index')
def index():
    pass


if __name__ == '__main__':
    app.run('0.0.0.0', 8000)
