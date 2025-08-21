import logging
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Initialize Flask app
application = Flask(__name__)
app = application

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO if you want less detail
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)

# Home page route
@app.route('/')
def home():
    logging.info("Rendering landing page: index.html")
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        try:
            # Log form data received
            logging.debug(f"Form Data Received: {request.form.to_dict()}")

            # Create CustomData instance from form inputs
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            logging.info("CustomData object created successfully.")

            # Convert input data to DataFrame
            data_frame = data.get_data_as_dataframe()
            logging.debug(f"DataFrame for prediction:\n{data_frame}")

            # Load prediction pipeline and make prediction
            predict_pipeline = PredictPipeline()
            logging.info("Initialized PredictPipeline.")

            results = predict_pipeline.predict(data_frame)
            logging.info(f"Prediction results: {results}")

            # Return home.html with prediction result
            return render_template('home.html', results=results[0])

        except Exception as e:
            logging.error("Error occurred during prediction", exc_info=True)
            return render_template('home.html', error=str(e))

    else:
        logging.info("GET request to /predict — rendering home.html without results.")
        return render_template('home.html')
    

if __name__ == "__main__":
    logging.info("Starting Flask application on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)



