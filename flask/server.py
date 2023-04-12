import random
import os
import yaml
from flask import Flask, request, jsonify
from keyword_spotting_service import kss_factory

config = yaml.safe_load(open('config.yml'))
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    # get audio file
    audio_file = request.files["file"]

    # save audio file
    file_name = str(random.randint(0, 100000))
    file_path = os.path.join(config['upload'], file_name)
    audio_file.save(file_path)

    # invoke keyword spotting service
    kss = kss_factory(os.path.join(config['model']))

    # perform inference
    prediction = kss.predict(file_path)

    # remove audio file
    os.remove(file_path)

    # return prediction as json object
    return jsonify({"keyword": prediction})

if __name__ == "__main__":
    app.run(debug=False)