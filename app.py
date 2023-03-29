import os
import io
import base64
from flask import Flask, render_template, request, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import numpy
import keras_ocr
from pdf2image import convert_from_path
import PyPDF2
import cv2
UPLOAD_FOLDER = 'PDFs/'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "We Love IA"
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/preprocessing', methods=['POST'])
def preprocessing():
    returnedName = ""
    fileName = request.form.get('fileName')
    if(fileName == None):
        return [False]
    Image.MAX_IMAGE_PIXELS = 1000000000 
    #Size of the log
    log_width = 2048
    log_height = 2048

    #Imports for OCR by Keras

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #Pipeline initialization
    pipeline = keras_ocr.pipeline.Pipeline()

    #Get the folder with the pdf in the sub directory
    #folder = os.listdir(main_path + "/" + sub_dir)
    folder = os.path.join(app.config['UPLOAD_FOLDER'],fileName.split('.')[0])

    #Verify that the pdf isn't already scanned
    if "LOGS" not in folder :
            
        print ("\nSearching in document : " + fileName)
        
        #Open the PDF
        pdf_reader = PyPDF2.PdfReader(os.path.join(folder, fileName), strict=False)
        #For each page of the PDF
        for page_num in range(len(pdf_reader.pages)):
            #Get pages one by one
            page = pdf_reader.pages[page_num]
        
            #Get the size of the page
            page_size = page.mediabox.upper_right

            #Check if the size of the page is over 4000
            if page_size[1] > 4000:
                
                #If the page height is over 4000 take it from the original PDF and convert it into image
                pages = convert_from_path(os.path.join(folder, fileName), first_page=page_num+1, last_page=page_num+1, thread_count=8)
                image = pages[0]
                
                #Get the width and height of the new image
                img_width, img_height = image.size
                
                #Get the size of the image and crop it at A4 format
                image_test = image.crop(( ((img_width // 2) - (log_width // 2)) - 120 , 0, ((img_width // 2) + (log_width // 2)) - 120, 1000))
                
                print("\n")
                
                # Use keras_ocr to take the text from the image
                results = pipeline.recognize([numpy.array(image_test)])

                text = []

                #Collect only the text from all pipeline.recognize informations
                for result in results[0]:
                    text.append(result[0])
                print(text)
                #If the choosen words are in the image
                if ( ("completion" in text and "log" in text) or ("composite" in text and "log" in text) or ("composite" in text and "core" in text) ):
                    
                    #Initialization of necessary variables
                    output_images = []
                    total_height = 0
                    
                    #While all the completion log isn't all croped
                    while total_height < img_height :
                        
                        #Crop a new image juste under the last one 
                        image_output = image.crop(( ((img_width // 2) - (log_width // 2)) - 120 , total_height, ((img_width // 2) + (log_width // 2)) - 120, total_height + log_height))
                        
                        #Add new croped image to the image list and increment the total_height
                        output_images.append(image_output)
                        total_height = total_height + 2048
                        
                    #Count nume of croped image to name files diferently
                    output_num = 1
                    
                    os.mkdir(os.path.join(folder, 'LOGS'))
                    
                    #For all croped images
                    
                    for output in output_images:
                        
                        # Convert the image to a numpy array
                        img_array = cv2.cvtColor(numpy.array(output), cv2.COLOR_RGB2BGR)

                        # Save the image as a JPEG file
                        image_name = os.path.basename(fileName)
                        if(returnedName == ""):
                            returnedName = f'{image_name}_{page_num}_{output_num}.jpg'
                        cv2.imwrite(os.path.join(os.path.join(folder, 'LOGS')  , f'{image_name}_{page_num}_{output_num}.jpg'), img_array)
                        
                        #Increment
                        output_num = output_num + 1
    return [True, returnedName]


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
        p = os.path.join(app.config['UPLOAD_FOLDER'],filename.split('.')[0])
        if(not(os.path.isdir(p))):
            os.mkdir(p)
        file.save(os.path.join(p, filename))
        return [True, filename]
    return [False]

@app.route('/process', methods=['POST'])
def process():
    model = YOLO("best.pt")
    fileName = request.form.get('fileName')
    if(fileName==None):
        return [False]
    nameSplit = fileName.split('.')
    img = 'PDFs/'+nameSplit[0]+'/LOGS/'+fileName
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
        if(bestClasses[i] != []):
            pilImage = im1.crop((bestClasses[i][0][0],bestClasses[i][0][1],bestClasses[i][0][2],bestClasses[i][0][3]))
            buffered = io.BytesIO()
            pilImage.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            cropedImages[i] = img_str
    print(cropedImages)

    return [True, cropedImages]

if __name__ == '__main__':
   app.run()