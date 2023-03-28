import os
import io
import base64
from flask import Flask, render_template, request, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import numpy

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "We Love IA"
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return [False]
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return [False]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return [True, filename]
    return [False]

@app.route('/process', methods=['POST'])
def process():
    model = YOLO("best.pt")
    fileName = request.form.get('fileName')
    img = 'uploads/'+fileName
    if(fileName==None):
        return [False]
    im1 = Image.open(img)
    results = model.predict(source=im1)
    nbrClasses = len(results[0].names)
    bestClasses = [[]]*nbrClasses
    cropedImages = {}
    for r in results:
        r = r.numpy()
        for i in range(len(r.boxes.xyxy)):
            classDetected =int(r.boxes.cls[i]) 
            if(bestClasses[classDetected] != []):
                if(r.boxes.conf[i]>bestClasses[classDetected][1]):
                    bestClasses[classDetected] = [r.boxes.xyxy[i],r.boxes.conf[i]]
            else:
                bestClasses[classDetected] = [r.boxes.xyxy[i],r.boxes.conf[i]]
    for i in range(len(bestClasses)):
        pilImage = im1.crop((bestClasses[i][0][0],bestClasses[i][0][1],bestClasses[i][0][2],bestClasses[i][0][3]))
        buffered = io.BytesIO()
        pilImage.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        cropedImages[i] = img_str
    print(cropedImages)

    return [True, cropedImages]

if __name__ == '__main__':
   app.run()