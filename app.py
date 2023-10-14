import os

from flask import Flask, request,render_template
import tensorflow as tf
import numpy as np 
from PIL import Image
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods=['POST','GET'])

def function():
    brands= ['adidas', 'converse', 'nike']
    img_height = 180
    img_width = 180
    model = tf.keras.models.load_model('final_model/model')
    data= request.files['data']
    print(data)

    img_path = os.path.abspath(os.path.join('static', 'uploads', data.filename))
    data.save(img_path)
    image = Image.open(data).resize((img_height, img_width))
    image = image.resize((img_height, img_width))

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    brand = brands[np.argmax(score)]
    score = round(100 * np.max(score),2)

    img_url = os.path.join('static', 'uploads', data.filename)

    logo_url = "../static/"

    if brand == 'converse':
        logo_url += "converse.png"
    elif brand == 'nike':
        logo_url += "nike.png"
    else:
        logo_url += "adidas.png"

    
    return render_template('index.html', score=score, brand=brand, img_url=img_url, logo_url=logo_url)