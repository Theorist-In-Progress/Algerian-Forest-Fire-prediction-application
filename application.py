import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 


application = Flask(__name__)
app=application

### import ridge regressor for algerian and scaler pickle
ridge_model=pickle.load(open('Models/algerian_forest.pkl','rb'))
standard_scaler=pickle.load(open('Models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Region = float(request.form.get('Region'))
        Classes= float(request.form.get('Classes'))
        
        new_data_scaled =standard_scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,ISI,Region,Classes]])
        results=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=results[0])

    else:
        return render_template('home.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0")