from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Define the model (same as the one used during training)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # EfficientNetV2 pre-trained model
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        # Adjust the final classification layer for 2 classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)

# Load the trained model
model = MyModel()
model_path = '../model/tumor.pth'  # Go up one level, then into the 'model' folder
model.load_state_dict(torch.load(model_path))  # Update with the correct path
model.eval()  # Set the model to evaluation mode

# Define the image transformation used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match EfficientNetV2 input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Same normalization as training
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image from the file
    img = Image.open(file.stream)

    # Preprocess the image
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability

    # Map the output to your labels (e.g., 0 = 'no', 1 = 'yes')
    label = 'yes' if predicted.item() == 1 else 'no'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Run on port 5001
