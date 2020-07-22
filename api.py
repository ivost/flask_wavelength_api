from flask import Flask, request , jsonify
from PIL import Image
import sys
import numpy as np
import cv2
import base64
import requests


sys.path.append('/usr/local/lib/python3.6/dist-packages/')

URL = "http://172.31.77.241:8080/predictions/fastrcnn"

app = Flask(__name__)
app.config["DEBUG"] = True

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

@app.route('/')
def hello_world():
   return 'Flask API Server'

@app.route('/api/1.0/classify', methods=['POST'])
def classify():
    # set the file as the data of the post request to our inference API

    try:
        data = request.files['file']
    except:
        response = jsonify({"Status": 500, "Message": "Error retrieving file."})
        return response

    image = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # send the inference reply
    r=requests.post(URL, data=data)

    # receive the response from the inference engine
    data = r.json()

    # extract the coordinates and label from the returned data
    results = data[0]
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

    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "png")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    response = jsonify({ "Status": 200,
        "Message": "Ok",
        "Image": str(img_base64)
        })

    return response

app.run(host="0.0.0.0")