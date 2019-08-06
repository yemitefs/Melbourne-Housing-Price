from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json
gbr = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def make_prediction():
     data = request.get_json(force=True)
     #convert our json to a numpy array
     one_hot_data = input_to_one_hot(data)
     predict_request = gbr.predict([one_hot_data])
     output = [predict_request[0]]
     print(data)
     return jsonify(results=output)

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(14)
    # set the numerical input as they are
    enc_input[0] = data['Rooms']
    enc_input[1] = data['Type']
    enc_input[2] = data['Method']
    enc_input[3] = data['Distance']
    enc_input[4] = data['Bedroom2']
    enc_input[5] = data['Bathroom']
    enc_input[6] = data['Car']
    enc_input[7] = data['Landsize']
    enc_input[8] = data['CouncilArea']
    enc_input[9] = data['Lattitude']
    enc_input[10] = data['Longtitude']
    enc_input[11] = data['Regionname']
    enc_input[12] = data['Propertycount']
    enc_input[13] = data['month']

    return enc_input


@app.route('/result',methods=['POST'])
def get_delay():
    result=request.form
    Rooms = result['Rooms']
    Type = result['Type']
    Method = result['Method']
    Distance = result['Distance']
    Bedroom2 = result['Bedroom2']
    Bathroom = result['Bathroom']
    Car = result['Car']
    Landsize = result['Landsize']
    CouncilArea = result['CouncilArea']
    Lattitude = result['Lattitude']
    Longtitude = result['Longtitude']   
    Regionname = result['Regionname']
    Propertycount = result['Propertycount']
    month = result['month']

    user_input = {'Rooms':Rooms, 'Type':Type, 'Method':Method, 'Distance':Distance,
               'Bedroom2':Bedroom2, 'Bathroom':Bathroom,'Car':Car,'Landsize':Landsize,
               'CouncilArea':CouncilArea,'Lattitude':Lattitude,'Longtitude':Longtitude,
               'Regionname':Regionname,'Propertycount':Propertycount, 'month':month}
     
    print(user_input)
    a = input_to_one_hot(user_input)
    price_pred = gbr.predict([a])[0]
    price_pred = round(price_pred, 2)    
    return json.dumps({'price':price_pred});




if __name__ == '__main__':
    app.run(port=4000, debug=True)
