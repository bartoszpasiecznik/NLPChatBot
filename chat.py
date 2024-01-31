import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def convert_temp_cel(kelvin):
    celsius = kelvin - 273.15
    return round(celsius)


def weather_info(city):
    api_key = "API_ID"
    city_name = city
    country_code = 'PL'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name},{country_code}&appid={api_key}'

    response2 = requests.get(url)

    if response2.status_code == 200:
        weather_data = response2.json()
        if len(weather_data) > 0:
            city = weather_data['name']
            temp_celsius = convert_temp_cel(weather_data['main']['temp'])
            formatted_celsius = f'{temp_celsius}°C'
            description = weather_data['weather'][0]['description']
            humidity = weather_data['main']['humidity']
            formatted_humidity = f'{humidity}%'
            wind_speed = weather_data['wind']['speed']
            formatted_wind = f'{wind_speed}m/s'

            city_details = {
                'city' : city,
                'celsius' : formatted_celsius,
                'description' : description,
                'humidity' : formatted_humidity,
                'Wind Speed' : formatted_wind
            }
            return city_details
        else:
            print(f"No results found for {city}")
    else:
        print(f"Failed to retrieve data for {city}")


print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    elif "weather" in sentence:
        search_city = input("In which city?: ")
        weather = weather_info(search_city)
        if weather:
            response2 = f"Here are the weather details for {search_city}\n"
            response2 += f"City: {weather['city']}\n"
            response2 += f"Temperature: {weather['celsius']}\n"
            response2 += f"Weather Description: {weather['description']}\n"
            response2 += f"Humidity: {weather['humidity']}\n"
            response2 += f"Wind Speed: {weather['Wind Speed']}"

            print(response2)

    else:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")