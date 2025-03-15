import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv

# Get the route to the dataset
data_dir = os.getenv('CEC_2025_dataset')

# same type of image transformation that it was trained on
# same size, same channels, change to RGB for how model likes it
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pick the small model with the imag network
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1") 
# again we want binary output at our last layer
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  
# load this model to use it
model.load_state_dict(torch.load("final_model.pth"))
# add to device
model.to(device)
# evaluate
model.eval()  

# some classifcations 
def classify_probability(prob):
    if prob < 0.5:
        return "Very Unlikely"
    elif prob < 0.75:
        return "Unlikely"
    elif prob < 0.9:
        return "Likely"
    else:
        return "Very Likely"
    
def predict_image(image_path):
    # open the image
    image = Image.open(image_path)
    # transform the image
    image = transform(image).unsqueeze(0)
    # move it to the device
    image = image.to(device)

    # get the output of the model
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        # return the prediction and confidence of it
        return predicted.item(), confidence.item()

def test_images_in_order(root_dir):
    # for each image in out CEC_test folder
    cec_test_folder = os.path.join(root_dir, "CEC_test")
    cec_test_images = sorted(
        [os.path.join(cec_test_folder, f) for f in os.listdir(cec_test_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    )  # Ensures images are processed in order

    # counters
    total = 0
    confidence_scores = []

    # output file name
    output_file = "output.csv"
    # open the output file
    with open(output_file, mode='w', newline='') as file:
        # create the columns
        writer = csv.writer(file)
        writer.writerow(["File Name", "Prediction", "Confidence", "Probability"])
        
        # for each image place the data accoridngly
        for image_path in cec_test_images:
            predicted, confidence = predict_image(image_path)
            prediction_label = "Yes" if predicted == 1 else "No"
            probability_label = classify_probability(confidence)
            writer.writerow([os.path.basename(image_path), prediction_label, f"{confidence:.2f}", probability_label])
            confidence_scores.append(confidence)
            total += 1
    
    # print some stats at the end
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    print(f"\nTested {total} images.")
    print(f"Average Confidence Score: {avg_confidence:.2f}")
    print(f"Results saved to {output_file}")
    
# check if directory exists if not don't do anything
if data_dir:
    print(f"Using dataset directory: {data_dir}")
    test_images_in_order(data_dir)
else:
    print("Environment variable CEC_2025_dataset is not set. Please set it to the path of your dataset.")
