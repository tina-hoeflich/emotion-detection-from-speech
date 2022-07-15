from cgitb import reset
import os
import pickle
from flask import Flask, jsonify,request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
import numpy as np
from prepare import load_x
from mimetypes import guess_extension
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Require a parser to parse our POST request.
# Unpickle our model so we can use it!
if os.path.isfile("./our_model.pkl"):
  model = pickle.load(open("./our_model.pkl", "rb"))
else:
  raise FileNotFoundError

import os
from flask import abort, current_app, make_response, request


@app.route('/predict', methods=['POST'])
def post():
  if(request.method=='POST'):
    response = jsonify(message="POST request returned")
    response.headers.add("Access-Control-Allow-Origin", "*")
    data = request.files['audio_data'].read()
    x_unknown = load_x(data)
    _y = model.predict(x_unknown)[0]
    response = {"class": _y}
    return response


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')



# api.add_resource(Predict, "/predict")
if __name__ == "__main__":
  app.run(debug=True)