import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from flask import jsonify
#import jsonify

import pickle
#from sklearn.externals import joblib
import pandas as pd
import cv2
import numpy as np
from skimage import feature

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])

def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('uploaded_file',
                                filename=filename))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #global model, mask
    print "uploads/" + filename
    image = cv2.imread("uploads/" + filename)
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    Cutted = cv2.resize(gray, (500, 500))
    (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
    H = np.array(H).reshape((1, -1))

    #prediction = clf.predict('/uploads/<filename>')
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    prediction = model.predict(H)
    print prediction
    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    #clf = joblib.load('../mlOffice/Linear_Office.txt')
    model = pickle.load(open('../mlOffice/Linear_Office.txt', 'rb'))

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

