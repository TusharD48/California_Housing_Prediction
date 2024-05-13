import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash,session,escape

import numpy as np


app = Flask(__name__)
# Load the Model
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_Data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    ouput = regmodel.predict(new_Data)
    print(ouput[0])
    return jsonify(ouput[0])

if __name__ == "main":
    app.run(debug=True)
    