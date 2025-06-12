import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load("model/trashnet_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)
    model = load_model()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return LABELS[predicted.item()]
