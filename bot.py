import torch
from torchvision import transforms, models
from PIL import Image

target_size = (256,256)

model = torch.load('trained_model.pth', weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path)

    width, height = image.size
    sides_ratio = target_size[0] / target_size[1]

    if width / height > sides_ratio:
        new_width = int(height * sides_ratio)
        left = (width - new_width) / 2
        right = left + new_width
        image = image.crop((left, 0, right, height))
    else:
        new_height = int(width / sides_ratio)
        top = height - new_height
        bot = top + new_height
        image = image.crop((0, top, width, bot))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

image_path = '1.jpg'
input_image = preprocess_image(image_path)

with torch.no_grad():
    output = model(input_image)
    probabilities = torch.softmax(output, dim=1)[0]

top_probs, top_classes = torch.topk(probabilities, 3)
print("Top 3 predictions:")
for i in range(3):
    print(f"Class {top_classes[i].item()}: {top_probs[i].item() * 100:.2f}%")

_, predicted_class = torch.max(output, 1)
print(f"\nPredicted class: {predicted_class.item()}")