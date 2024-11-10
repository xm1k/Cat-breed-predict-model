import torch
from graphviz.backend.dot_command import command
from torchvision import transforms, models
from PIL import Image
import telebot
from config import TOKEN

target_size = (256,256)

model = torch.load('trained_model.pth', weights_only=False)
model.eval()

is_cat_model = torch.load('./is_cat/is_cat_model.pth', weights_only=False)
is_cat_model.eval()

cat_names = [
    "Абиссинская",
    "Ангорская",
    "Балинезийская",
    "Бенгальская",
    "Бомбейская",
    "Британская короткошёрстная",
    "Бирманская",
    "Шартрез",
    "Европейская короткошёрстная",
    "Японский бобтейл",
    "Корат",
    "Мейн-кун",
    "Невская маскарадная",
    "Норвежская лесная",
    "Персидская",
    "Рэгдолл",
    "Рекс",
    "Русская голубая",
    "Саванна",
    "Шотландская вислоухая",
    "Сиамская",
    "Сингапурская",
    "Сфинкс"
]

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

###

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, """Привет! 👋 Я - твой помощник по распознаванию кошек!
Отправь мне своего пушистого друга🐱 и я расскажу тебе, какая это порода 🐈
Для лучшего результата, отправляй фотографии, где кошка хорошо видна.""")
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = "received_photo.jpg"
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    input_image = preprocess_image(image_path)

    bot.send_message(message.chat.id, "Анализ 🔎")
    with torch.no_grad():
        is_cat_output = torch.softmax(is_cat_model(input_image), dim=1)
        cat_probability = is_cat_output[0, 1].item()

        if cat_probability < 0.5:
            response = "На фото не обнаружена кошка или кошку плохо видно😿. Пожалуйста, отправьте другое изображение."
        else:
            output = model(input_image)
            probabilities = torch.softmax(output, dim=1)[0]
            top_probs, top_classes = torch.topk(probabilities, 3)
            predictions = []
            for i in range(3):
                probability = top_probs[i].item() * 100
                predictions.append(f"{cat_names[top_classes[i].item()]}: {probability:.2f}%")

            response = "Вероятные породы:\n" + "\n".join(predictions)
    bot.send_message(message.chat.id, response)

bot.polling(none_stop=True)


###

# Обработка локальных фотографий:

# image_path = '1.jpg'
# input_image = preprocess_image(image_path)
# with torch.no_grad():
#     output = model(input_image)
#     probabilities = torch.softmax(output, dim=1)[0]
#
# top_probs, top_classes = torch.topk(probabilities, 3)
# print("Top 3 predictions:")
# for i in range(3):
#     print(f"{cat_names[top_classes[i].item()]}: {top_probs[i].item() * 100:.2f}%")
#
# _, predicted_class = torch.max(output, 1)
# print(f"\nPredicted class: {predicted_class.item()}")