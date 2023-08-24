import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('./model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]  
    result = str(features)
    prediction = model.predict(features) 
    if prediction[0] == 0:
        result = "Prediction: Graduate"
    else:
        result = "Prediction: Dropout"    

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)