# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:47:26 2020

@author: Sharjeel Masood
"""

import csv
import numpy as np
import cv2

raw_data = []

with open("C:\\Users\\Dutchfoundation\\Desktop\\FYP\\nano-degree simulator\\driving_log.csv", 'r') as csv_file:
    
    csv_reader = csv.reader(csv_file)
    
    for line in csv_reader:
        raw_data.append(line)
    
data = np.array(raw_data)

training_data = []

for i in data:
    center = i[0]
    left = i[1]
    right = i[2]
    steering_angle = i[3]
    
    # for center image
    image = cv2.imread(center)#, 0)
    image = cv2.resize(image, (80, 80))
    st = steering_angle
    training_data.append([np.array(image), float(st)])
    
    # for left
    left_image = cv2.imread(left)#, 0)
    left_image = cv2.resize(left_image, (80, 80))
    st = float(steering_angle) + 0.15
    training_data.append([np.array(left_image), st])
    
    # for right
    right_image = cv2.imread(right)#, 0)
    right_image = cv2.resize(right_image, (80, 80))
    st = float(steering_angle) - 0.15
    training_data.append([np.array(right_image), st])
    
training_data = np.array(training_data)

# ---------Balancing the dataset--------------
left = []
right = []
straight = []

for sample in training_data:
    angle = sample[1]
    
    if float(angle) < 0:
        left.append(sample)
    elif float(angle) > 0:
        right.append(sample)
    elif float(angle) == 0:
        straight.append(sample)

left = np.array(left)
right = np.array(right)
straight = np.array(straight)

maximum = min(len(left), len(right), len(straight))

left = left[:maximum]
right = right[:maximum]
straight = straight[:800]

print('left: ', left.shape)
print('right: ', right.shape)
print('straight: ', straight.shape)

final_data = []

for li in left:
    final_data.append(li)
for ri in right:
    final_data.append(ri)
for si in straight:
    final_data.append(si)

final_data = np.array(final_data)
print('\n', 'Total: ', len(final_data))
print(final_data.shape)

#for sample in right:
 #   print(sample[1])

np.random.shuffle(final_data)
np.save('new_test_data_rgb.npy', final_data)
print('File saved...')
