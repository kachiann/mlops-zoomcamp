# Deployment in batch mode

This project focuses on deploying the ride duration model in batch mode using the Yellow Taxi Trip Records dataset.

## We will be:

- Creating a virtual environment with Pipenv
- Creating a script for prediction
- Putting the script into a Flask app
- Packaging the app to Docker

### Overview

The goal of this project is to deploy a machine learning model that predicts ride durations based on the Yellow Taxi Trip Records dataset. We will follow these steps:

1. **Setting Up Environment**
   - Create a virtual environment using Pipenv to manage dependencies.

2. **Script for Prediction**
   - Develop a Python script (`apply_model.py`) that reads trip data, applies a pre-trained model to predict ride durations, and saves results.

3. **Integration with Flask**
   - Incorporate the prediction script into a Flask web application to expose predictions via a RESTful API.

4. **Packaging with Docker**
   - Containerize the Flask application along with its dependencies into a Docker image for scalability and deployment.
