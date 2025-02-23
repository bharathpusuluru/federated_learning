from flask import Flask, request, send_file, jsonify
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Storage folders
SCRIPT_FOLDER = "scripts/"
MODEL_FOLDER = "models/"
os.makedirs(SCRIPT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Path to the training script
SCRIPT_PATH = SCRIPT_FOLDER + "main.py"

@app.route('/')
def home():
    return '''
    <h1>Federated Learning Server</h1>
    <p><a href="/download_script">Download Training Script</a></p>
    <h2>Upload Trained Model</h2>
    <form action="/upload_model" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".h5">
        <input type="submit" value="Upload">
    </form>
    <h2>Uploaded Models</h2>
    <a href="/list_models">View Uploaded Models</a><br><br>
    <h2>Model Aggregation</h2>
    <form action="/average_models" method="post">
        <input type="submit" value="Average Uploaded Models">
    </form>
    '''

@app.route('/download_script', methods=['GET'])
def download_script():
    return send_file(SCRIPT_PATH, as_attachment=True)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    
    if not file.filename.endswith(".h5"):
        return "Only .h5 files are allowed", 400

    file_path = os.path.join(MODEL_FOLDER, file.filename)
    file.save(file_path)
    return "Model uploaded successfully!", 200

@app.route('/list_models', methods=['GET'])
def list_models():
    models = os.listdir(MODEL_FOLDER)
    return jsonify({"uploaded_models": models})

@app.route('/average_models', methods=['POST'])
def average_models():
    model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.h5')]
    
    if len(model_files) < 2:
        return "At least two models are required for averaging!", 400

    # Load all models
    models = [tf.keras.models.load_model(os.path.join(MODEL_FOLDER, f)) for f in model_files]
    weights = [model.get_weights() for model in models]

    # Ensure all models have the same structure
    for i in range(1, len(weights)):
        assert len(weights[i]) == len(weights[0]), "Model structures do not match!"

    # Average the weights
    averaged_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]

    # Create new model and set averaged weights
    averaged_model = tf.keras.models.clone_model(models[0])
    averaged_model.set_weights(averaged_weights)

    # Save the averaged model
    averaged_model_path = os.path.join(MODEL_FOLDER, "averaged_model.h5")
    averaged_model.save(averaged_model_path)

    return "Averaged model saved as 'models/averaged_model.h5'", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
