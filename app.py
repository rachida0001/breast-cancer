from flask import Flask, request , render_template
import joblib
import numpy as np

#Create Flask app
app = Flask(__name__)

#moad the model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        # Preprocess the data as needed
        prediction = model.predict(features)
        if prediction==1:
            txt = "benign"
        else:
            txt = "malignant"
        return render_template('index.html', prediction_text ="the breast cancer is {}".format(txt))

if __name__ == '__main__':
    app.run(debug=True)