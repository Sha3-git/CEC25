from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Define the model architecture (same as training)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)  # No pretrained weights
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)

# Load the model
model = MyModel()

# Load the trained model weights
model_path = os.path.join(os.getcwd(), 'model', 'model_tumor_model_100_A.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match EfficientNetV2 input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load and preprocess the image
        img = Image.open(file.stream)
        img = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Get the probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get the max probability and predicted class

        # Map the result
        label = 'yes' if predicted.item() == 1 else 'no'

        return jsonify({'prediction': label, 'confidence': confidence.item()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
