from flask import Flask, request , jsonify
from PIL import Image
import sys, io
import numpy as np
import cv2
import base64
import requests
import boto3

# takes the coordinates as a string and returns a pair of tuples
def str_to_tup(mystring):
    i = 0
    s = mystring

    return_list = []
    l = s.split('), (')
    for item in l:
        temp_list = item.strip('[]()').split(',')
        for number in temp_list:
            temp_list[i % 2] = number.strip('. ').split('.')[0]
            i=i+1
        return_list.append(tuple(temp_list))
    return return_list

app = Flask(__name__)
app.config["DEBUG"] = True

sys.path.append('/usr/local/lib/python3.6/dist-packages/')

try:
    URL = open("config_values.txt",'r').readline().split('\n')[0]
except:
    print("Error reading configuration file")
    sys.exit(1)

@app.route('/')
def hello_world():
   return 'Flask API Server'

@app.route('/api/1.0/classify', methods=['POST'])
def classify():
    try:
        data = request.files['file']
    except:
        response = jsonify({"Status": 500, "Message": "Error retrieving file."})
        return response

    image = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # send the inference reply
    data.seek(0)
    r=requests.post(URL, data=data)

    # receive the response from the inference engine
    response = r.json()

    print(response)

    if response['code']:
        print (response['code'])
    else:
        # extract the coordinates and label from the returned data
        results = response[0]
        for result in results:
            coordinates = str_to_tup(results[result])
            label = result

        position = (10,50)

        cv2.putText(
            image, #numpy array on which text is written
            label, #text
            position, #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            1, #font size
            (209, 80, 0, 255), #font color
            3) #font stroke

        start_point = (int(coordinates[0][0]), int(coordinates[0][1]))
        end_point = (int(coordinates[1][0]), int(coordinates[1][1]))

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(image, start_point, end_point, color, thickness)

        s3 = boto3.resource('s3')
        object = s3.Object('my_bucket_name', 'my/key/including/filename.txt')
        object.put(Body=img)

        img = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "png")
        rawBytes.seek(0)
        img_base64 = str(base64.b64encode(rawBytes.read()))

        response = jsonify({ "code": 200,
            "message": "OK",
            "image": img_base64
            })

        return response

app.run(host="0.0.0.0")