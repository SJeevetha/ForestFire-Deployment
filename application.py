import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  

application = Flask(__name__)
app=application

## import ridge regressor and standar scalerpickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        Rh = float(request.form.get('RH'))
        Ws = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform(
            [[Temperature, Rh, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(debug=True)