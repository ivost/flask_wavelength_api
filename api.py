from flask import Flask, request , jsonify
from PIL import Image
import numpy as np
import sys, io, cv2, base64, requests, ast


app = Flask(__name__)
app.config["DEBUG"] = True

def parse_coordinates(s):
    """
    Parses a string with box'es coordinates into
    actual coordinate tuples.

    Example:
        >>> parse_coordinates("[(228.7825, 82.63463), (583.77545, 677.3058)]")
        ((228, 82), (583, 677))
    """
    box = ast.literal_eval(s)
    return tuple(tuple(int(v) for v in coord) for coord in box)

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

    # convert response to json
    response = r.json()
    # print response to console
    print('Response received from inference server at: ' + URL + ': ')
    print(response) 

    # if response has a key named 'code' we have an error otherwise with have a list of objects detected in the image
    if 'code' in response:
        return response
    else:
        # extract the coordinates and label of the first detected object from the returned data
        results = response[0]
        label, coordinates = next(iter(results.items()))
        coordinates = parse_coordinates(coordinates)
        
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

        #set the coordinates for the bounding box
        start_point, end_point = coordinates

        # Box clolor will be blue
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Draw a rectangle with blue borders of thickness of 2 px
        img = cv2.rectangle(image, start_point, end_point, color, thickness)

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