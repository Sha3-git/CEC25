import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

data_dir = os.getenv('CEC_2025_dataset')

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1") 
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  
model.load_state_dict(torch.load("tumor_model_1000.pth"))
model.to(device)
model.eval()  

def classify_probability(prob):
    if prob < 0.25:
        return "Very Unlikely"
    elif prob < 0.5:
        return "Unlikely"
    elif prob < 0.75:
        return "Likely"
    else:
        return "Very Likely"
    
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        return predicted.item(), confidence.item()

def test_random_images(root_dir):
    cec_test_folder = os.path.join(root_dir, "CEC_test")
    cec_test_images = [os.path.join(cec_test_folder, f) for f in os.listdir(cec_test_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    total = 0
    confidence_scores = []

    output_file = "output.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Prediction", "Confidence", "Probability"])
        
        for image_path in cec_test_images:
            predicted, confidence = predict_image(image_path)
            prediction_label = "Yes" if predicted == 1 else "No"
            probability_label = classify_probability(confidence)
            
            writer.writerow([os.path.basename(image_path), prediction_label, f"{confidence:.2f}", probability_label])
            confidence_scores.append(confidence)
            total += 1
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    print(f"\nTested {total} images.")
    print(f"Average Confidence Score: {avg_confidence:.2f}")
    print(f"Results saved to {output_file}")
    
if data_dir:
    print(f"Using dataset directory: {data_dir}")
    test_random_images(data_dir)
else:
    print("Environment variable CEC_2025_dataset is not set. Please set it to the path of your dataset.")