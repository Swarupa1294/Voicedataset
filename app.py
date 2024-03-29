from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from features import extract_feature
app = Flask(__name__)

## Configure upload location for audio
app.config['UPLOAD_FOLDER'] = "./audio"

## Route for home page
@app.route('/')
def home():
    return render_template('index.html',value="")


## Route for results
@app.route('/result', methods = ['GET', 'POST'])
def results():

    if not os.path.isdir("./audio"):
        os.mkdir("audio")

    if request.method == 'POST':
        try:
          f = request.files['file']
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        except:
          return render_template('index.html', value="")

    wav_file  = os.listdir("./audio")[0]
    wav_file = f"{os.getcwd()}/audio/{wav_file}"
    model = pickle.load(open(f"{os.getcwd()}/result/Decisiontree-main.model", "rb"))
    x_test = extract_feature(wav_file)
    y_pred = model.predict(np.array([x_test])) 
    os.remove(wav_file)
    return render_template('predict.html', value= y_pred[0])
    print( y_pred)
if __name__ == "__main__":
    app.run(debug=True)