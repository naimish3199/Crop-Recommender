import numpy as np
from flask import Flask, request, render_template
import pickle
#creating flask app
app = Flask(__name__,template_folder='template', static_folder='static')
model = pickle.load(open('model.pkl', 'rb'))
# rendering home.html
@app.route('/')
def home():
    return render_template('index.html')
# Post method provides features to model.pkl file so that it takes input and returns output
@app.route('/predict',methods=['POST'])
def predict():
    #
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)] # storing input values as numpy array
    prediction = model.predict(final_features) # predicting

    output = prediction[0]

    return render_template('index.html', prediction_text='Best Crop to Grow is {}'.format(output))



if __name__ == "__main__": # runs complete flask
    app.run(debug=True)
    
    