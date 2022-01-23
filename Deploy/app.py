import numpy as np
import cv2
import os
from utils import *
from flask import Flask, request, render_template, redirect
from keras.models import load_model

model = load_model(MODEL_PATH)

# Initialize
if os.path.exists("static/save/last.jpg"):
    os.remove("static/save/last.jpg")
else:
    print("The file does not exist")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 #16KB

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print('No file')
        if os.path.exists("static/save/last.jpg"):
            os.remove("static/save/last.jpg")
        else:
            print("The file does not exist")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print('Empty file!')
        if os.path.exists("static/save/last.jpg"):
            os.remove("static/save/last.jpg")
        else:
            print("The file does not exist")
        return redirect(request.url)
    print('load successfully')
    if file:
        print(file)
        img_str = file.read()
        nparr = np.frombuffer(img_str, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('static/save/last.jpg', image)
        image = np.asarray(image)
        image = image.transpose()
        image = np.flip(image, axis=1)
        image = image/255.0 # Normalize
        pred = model.predict(image.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1))
        output = decode_output(pred)
        return render_template('index.html', prediction_text=f'{output}')

if __name__ == "__main__":
    app.run(debug=True)