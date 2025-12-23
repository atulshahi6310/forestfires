import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from flask import Flask, jsonify, request, render_template


application = Flask(__name__)
app = application
## importing ridge model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=[ 'get' ,'POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        X = np.array([[temperature, RH, Ws, Rain, FFMC, DMC, Classes, ISI, Region]])
        X_scaled = scaler.transform(X)
        result = ridge_model.predict(X_scaled)
        return render_template('home.html', result= result[0])
    else:
        return render_template('home.html') 






if __name__ == "__main__":
    app.run(debug=True ,host='0.0.0.0')