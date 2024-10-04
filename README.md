# Assessing-Impact-of-Digital-Lens-Usage-on-Eye-Dryness-using-Schirmer-s-Effect

## Overview
This project explores the impact of digital lens usage on eye dryness, leveraging machine learning models to predict eye dryness based on user inputs such as screen usage and lens type. Using Schirmer's Effect as a measurement for dryness, the project integrates multiple regression and classification models into a Flask web application for easy interaction and prediction visualization.

## Introduction
Assessing Impact of Digital Lens Usage on Eye Dryness using Schirmer's Effect aims to predict and assess eye dryness due to prolonged screen exposure and digital lens usage. The project uses machine learning models to analyze different factors contributing to eye dryness and offers insights on preventing digital eye strain.

## Features
Machine Learning Predictions: Predict eye dryness based on factors such as digital lens usage, screen time, and lighting conditions.
Model Comparison: Evaluate multiple machine learning models and compare their performance.
Flask Web Application: A user-friendly web interface that takes user input and returns predictions in real-time.
Visualization of Results: Display prediction results using visual graphs and charts.

## Tech Stack
Frontend: HTML, CSS
Backend: Flask, Python
Machine Learning Libraries: Scikit-learn, Pandas, Numpy
Models Used:
Logistic Regression
Decision Tree
Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor (SVR)
Decision Tree Regressor
Decision Tree Classifier (Best Performing Model)

## Installation
To run this project locally, follow these steps:

## Clone the repository:
git clone https://github.com/your-username/Assessing-Impact-of-Digital-Lens-Usage-on-Eye-Dryness.git

## Navigate to the project directory:
cd Assessing-Impact-of-Digital-Lens-Usage-on-Eye-Dryness

## Set up the environment:
Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows

## Install the required dependencies:
pip install -r requirements.txt

## Open the project in Spyder:
Open Spyder IDE.
Navigate to the project directory.
Open the **app.py** file in Spyder.

## Run the application:
In Spyder, you can run the **app.py** file directly by pressing the green **"Play"** button or by selecting Run > **Run**.
Make sure Flask is installed and working by checking the console output.

## Open the web app in your browser:
http://127.0.0.1:5000/ ## Copy this link and past in web browser

## Usage
Input Data: Users can input data related to digital lens usage, screen exposure time, and ambient conditions.
Model Selection: Users can choose between different models to predict eye dryness.
Results: After processing, the web app displays the prediction with visual graphs.
Model Evaluation: The app allows users to compare the performance of various models using metrics like accuracy, mean squared error, and more.

## Model Information
This project uses a range of machine learning models:
Logistic Regression: Used for binary classification, models the probability of an input belonging to a particular class using a logistic function.
Decision Tree: Splits the dataset recursively based on the most significant features and is interpretable for both numerical and categorical data.
Random Forest Regressor: An ensemble model that creates multiple decision trees and merges their predictions to reduce overfitting.
Decision Tree Regressor: Predicts continuous values by splitting data based on features to minimize variance.
Support Vector Regressor (SVR): A variation of SVM for regression, uses kernel functions to fit a linear model in higher-dimensional space.
Gradient Boosting Regressor: Sequentially builds weak learners (decision trees) that correct the errors of previous ones.
Decision Tree Classifier: Used for classification problems, splits data based on features to maximize information gain.

## Best Performing Model
The Decision Tree Classifier was the best performing model in this project. Its accuracy and ability to handle both numerical and categorical data made it the most suitable model for predicting eye dryness. It provided the highest precision and recall metrics, making it ideal for real-world use.

## Project Structure
Assessing-Impact-of-Digital-Lens-Usage-on-Eye-Dryness/
├── app.py                  # Main Flask application
├── templates/              # HTML, CSS files for the web interface
├── code/                   # Machine learning models
├── datasets/               # Datasets for training and testing
├── README.md               # Project documentation
├── requirements.txt        # Dependencies

## Future Scope
Incorporate More Lens Types: Include a wider range of lens types for more refined predictions.
Improve Model Accuracy: Experiment with other models such as neural networks or deep learning approaches for even better accuracy.
Real-Time Monitoring: Add features for real-time tracking of eye dryness using IoT devices or sensors.
