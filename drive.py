# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:26:54 2020

@author: Sharjeel Masood
"""

from Model_2 import Main_NN

import base64

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
import torch
from torchvision import transforms


sio = socketio.Server()
app = Flask(__name__)

speed_limit = 10

def transformations(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6473, 0.5195, 0.4208],
                             std=[0.1576, 0.0982, 0.0856])
        ])
    return transform(image)

@sio.on('telemetry')
def telemetry(sid, data):

    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.array(image)
    image = transformations(image).unsqueeze(0)

    
    with torch.no_grad():
        steering_angle = model(image)
    
    steering_angle = float(steering_angle.squeeze())
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        })


if __name__ == '__main__':

    model = Main_NN()
    state = torch.load('test_model_04.pth', map_location = 'cpu')
    model.load_state_dict(state)
    model.eval()
    
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)