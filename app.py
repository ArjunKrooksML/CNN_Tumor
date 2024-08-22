from flask import Flask, request, redirect, url_for, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import subprocess

app = Flask(__name__)
app.config['UPLOAD'] = 'uploads'

BTM = tf.keras.models.load_model('BrainTumorModel_VGG16_Final.h5')

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def popup(result):
    subprocess.run(['python', 'popup.py', result])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(f"{app.config['UPLOAD']}/{file.filename}")
        img_array = preprocess(f"{app.config['UPLOAD']}/{file.filename}")


        predictions = BTM.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]


        result = 'Tumor Detected' if predicted_class == 1 else 'No Tumor Detected'

        popup(result)
        return redirect(url_for('upload_file'))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)