from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset and train the model
path = "ObesityDataSet_raw_and_data_sinthetic.csv"
data = pd.read_csv(path)

# Label Encoding

le = LabelEncoder()
columns_to_encode = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
    "NObeyesdad",
]
for col in columns_to_encode:
    data[col + "_n"] = le.fit_transform(data[col])

# Prepare inputs and outputs
inputs = data.drop(
    [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
        "NObeyesdad",
        "NObeyesdad_n",
    ],
    axis=1,
)
outputs = data["NObeyesdad_n"]

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size=0.8)

# Standardize the inputs
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the model
model = SVC()
model.fit(x_train, y_train)

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form.get("Age", 0)),  # Default to 0 if missing
            float(request.form.get("Height", 0)),
            float(request.form.get("Weight", 0)),
            float(request.form.get("FCVC", 0)),
            float(request.form.get("NCP", 0)),
            float(request.form.get("CH2O", 0)),
            float(request.form.get("FAF", 0)),
            float(request.form.get("TUE", 0)),
            int(request.form.get("Gender_n", 0)),
            int(request.form.get("family_history_with_overweight_n", 0)),
            int(request.form.get("FAVC_n", 0)),
            int(request.form.get("CAEC_n", 0)),
            int(request.form.get("SMOKE_n", 0)),
            int(request.form.get("SCC_n", 0)),
            int(request.form.get("CALC_n", 0)),
            int(request.form.get("MTRANS_n", 0)),
        ]
    except ValueError:
        return "Invalid input: Please provide valid numeric values for all fields.", 400

    # Scale and predict
    scaled_data = sc.transform([data])
    prediction = model.predict(scaled_data)

    # Decode the prediction to a human-readable value
    prediction_decoded = le.inverse_transform(prediction)

    return render_template("result.html", prediction=prediction_decoded[0])


if __name__ == "__main__":
app.run(host="0.0.0.0", port=port)
