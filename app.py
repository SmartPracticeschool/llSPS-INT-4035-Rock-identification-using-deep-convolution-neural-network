import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
#model = load_model("mymodel.h5")
model_filepath = './mymodel.h5'

model = tf.keras.models.load_model(
    model_filepath,
    custom_objects=None,
    compile=False
)
global text

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            #2
            
            print("prediction",preds)
            
        index = ['Igneous','Metamorpic','Sedimentary',]
        
        text = "the predicted rock is : " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run()
        
        
        
    
    
    