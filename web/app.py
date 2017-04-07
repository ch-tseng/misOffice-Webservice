import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from flask import jsonify

#import pickle
from sklearn.externals import joblib
import pandas as pd
import cv2
import numpy as np
from skimage import feature

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])

def upload():
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print "uploads/" + filename
    image = cv2.imread("uploads/" + filename)
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    Cutted = cv2.resize(gray, (500, 500))
    (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
    H = np.array(H).reshape((1, -1))

    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    prediction = model.predict(H)
    print prediction[0]
    #return jsonify({'prediction': prediction[0]}) 
    return prediction[0] 

if __name__ == '__main__':
    model = joblib.load('../mlOffice/misoffice.pkl')
#    model = pickle.load(open('../mlOffice/Linear_Office.txt', 'rb'))

    mask = np.zeros((500,500), dtype="uint8")
    cv2.rectangle(mask, (1, 63), (174, 262), 255, -1)
    cv2.rectangle(mask, (1, 292), (180, 500), 255, -1)
    cv2.rectangle(mask, (300, 60), (480, 265), 255, -1)
    cv2.rectangle(mask, (300, 295), (480, 500), 255, -1)

    app.run(
        host="0.0.0.0",
        port=int("8080"),
        debug=True
    )

