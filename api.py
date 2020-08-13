from flask import Flask, request , jsonify
from PIL import Image
from flask_cors import CORS
import numpy as np
import sys, io, cv2, base64, requests, ast

app = Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True

try:
    # read the first line in the configuration file to get the URL for the inference server
    # line should be of the format: http://<server_ip_address>:<port>/predictions/<model_name>

    with open('config_values.txt', 'r') as f:
        for line in f:
            URL = line.strip()
            break
except:
    print('Error reading configuration file')
    sys.exit(1)

@app.route('/')
def hello_world():
   return 'Flask API Server'

@app.route('/api/1.0/classify', methods=['POST'])
def classify():
    try:
        data = request.files['file']
    except:
        response = jsonify({'code': 500, 'type': 'InternalServerException', 'message': 'Error retrieving file'})
        return response

    image = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # send the image to the infernce server and store the response in 'r'
    data.seek(0)
    r=requests.post(URL, data=data)
    
    # print response to console
    print('Response received from inference server at: ' + URL + ': ')
    response=r.json()
    print(response) 

    # if response has a key named 'code' we have an error otherwise with have a list of objects detected in the image
    if 'code' in response:
        return response
    else:
        # extract the coordinates and label of the first detected object from the returned data
        results = response[0]

        label=next(iter(results))
        top_x=int(results[label][0])
        top_y=int(results[label][1])
        bottom_x=int(results[label][2])
        bottom_y=int(results[label][3])

        # set the corners of the bounding rectangle
        start_point = (top_x, top_y)
        end_point = (bottom_x, bottom_y)
        
        # set the position where the text label will be written
        position = (10,50)

        #write text onto image
        cv2.putText(
            image,  
            label, 
            position, 
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            1, #font size
            (209, 80, 0, 255), #font color
            3) #font stroke

        # Box clolor will be blue
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Draw a rectangle with blue borders of thickness of 2 px
        rectimg = cv2.rectangle(image, start_point, end_point, color, thickness)
        img = cv2.cvtColor(rectimg, cv2.COLOR_BGR2RGB)

        # take the resulting image, convert it to png, and encode it in base64
        img = Image.fromarray(img.astype('uint8'))
        rawBytes = io.BytesIO()
        img.save(rawBytes, 'png')
        rawBytes.seek(0)
        img_base64 = str(base64.b64encode(rawBytes.read()))

        # format our response
        response = jsonify({ 'code': 200,
            'message': 'OK',
            'type': 'Success',
            'image': img_base64
            })

        # return the image, and response codes
        return response

app.run(host='0.0.0.0')