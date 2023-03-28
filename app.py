from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.files['file']
    # Do something with the file, like save it to disk
    return 'File uploaded successfully'

if __name__ == '__main__':
   app.run()