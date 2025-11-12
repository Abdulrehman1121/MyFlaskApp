import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__, template_folder='templates')

# Load model with absolute path
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as load_error:
    print(f"Error loading model: {load_error}")
    model = None

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # get values from form
            sepal_length = float(request.form["Sepal_Length"])
            sepal_width = float(request.form["Sepal_Width"])
            petal_length = float(request.form["Petal_Length"])
            petal_width = float(request.form["Petal_Width"])
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            output = model.predict(features)[0]
            return render_template("index.html", prediction_text=f"The Flower Name is {output}")
        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")
    else:
        # GET request
        return render_template("index.html")
