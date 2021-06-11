
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Pregnancies': 0,
                  'Glucose': 1, 'BloodPressure': 2, 'SkinThickness': 3, 'Insulin': 4,
                             'BMI': 5, 'DiabetesPedigreeFunction': 6, 'Age': 7})

print(r.json())
