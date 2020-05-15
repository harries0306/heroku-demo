import requests

url = 'http://localhost:8888/predict_api'
r = requests.post(url,json={'Daily Time Spent on Site':75, 'Age':30, 
                            'Area Income':700000,'Daily Internet Usage':250,'Male':0})

print(r.json())