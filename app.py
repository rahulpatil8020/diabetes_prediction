import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('NewLGBM.pkl')


def GetFinalFeatures(data):
    final = [0 for x in range(11)]
    final_num = [0 for x in range(13)]
    final[0] = 1 if data[7] <= 30 and data[1] <= 120 else 0
    final[1] = 1 if data[5] <= 30 else 0
    final[2] = 1 if data[7] <= 30 and data[0] <= 6 else 0
    final[3] = 1 if data[1] <= 105 and data[2] <= 80 else 0
    final[4] = 1 if data[3] <= 20 else 0
    final[5] = 1 if data[5] < 30 and data[3] <= 20 else 0
    final[6] = 1 if data[1] <= 105 and data[5] <= 30 else 0
    final[7] = 1 if data[4] < 200 else 0
    final[8] = 1 if data[2] < 80 else 0
    final[9] = 1 if data[0] < 4 and data[0] != 0 else 0
    final_num[0] = data[0]
    final_num[1] = data[1]
    final_num[2] = data[2]
    final_num[3] = data[3]
    final_num[4] = data[4]
    final_num[5] = data[5]
    final_num[6] = data[6]
    final_num[7] = data[7]
    final_num[8] = data[5] * data[3]
    final_num[9] = data[0] / data[7]
    final_num[10] = data[1] / data[6]
    final_num[11] = data[7] * data[6]
    final_num[12] = data[7] / data[4]
    final[10] = 1 if final_num[8] < 1034 else 0
    scaler = joblib.load('scaler.pkl')
    final_num_scaled = scaler.transform(np.array(final_num).reshape(1, -1))
    final = np.array(final).reshape(1, -1)
    final_scaled = np.concatenate((final, final_num_scaled), axis=1)
    return final_scaled


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = GetFinalFeatures(int_features)
    prediction = model.predict(final_features)
    output = 'Positive' if prediction[0] == 1 else 'Negative'
    return render_template('index.html', prediction_text='Your Diabetes Report is: {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
