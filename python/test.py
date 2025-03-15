import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import random

# nmber of image to test with
NUM_IMAGES = 1000

# similar as training to make consistent
# again, make tensors switch to RGB
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# load the device use cuda is possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use the small model for efficient net as its chosen
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

# make it a binary problem
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
# define model path
model.load_state_dict(torch.load("final_model.pth"))
# add to devile
model.to(device)
# make it eval mode to use
model.eval()

# lets test it now
def predict_image(image_path):
    # open image, and test it using the model return the value predicted
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# runner function
def test_random_images(root_dir, num_images=NUM_IMAGES):
    # define folder paths are we still test from them
    yes_folder = os.path.join(root_dir, "yes")
    no_folder = os.path.join(root_dir, "no")
    
    # get image filenames from 'yes' and 'no' folders
    yes_images = [os.path.join(yes_folder, f) for f in os.listdir(yes_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    no_images = [os.path.join(no_folder, f) for f in os.listdir(no_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # combine lists of images
    all_images = yes_images + no_images
    # shuffle them for fun
    random.shuffle(all_images)

    # limit the number of images to test (using our glob)
    all_images = all_images[:num_images]

    # running count
    correct = 0
    total = 0
    # for each image in images
    for image_path in all_images:
        # defined
        label = 1 if 'yes' in image_path else 0 
        predicted = predict_image(image_path)
        
        # print results
        prediction_label = "Yes" if predicted == 1 else "No"
        print(f"Image: {os.path.basename(image_path)} - Predicted: {prediction_label}")
        
        # if we got it right or wrong
        if predicted == label:
            correct += 1
        total += 1

    # print final stats
    accuracy = correct / total * 100
    print(f"\nTested {total} images.")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    
# runs tests in directory
data_dir = "CEC_2025"
test_random_images(data_dir)
