import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Read a sample image.
image = cv2.imread('media/sample.jpg')

def WhiteToBlack(img):
    white_to_black_table = []
    for i in range(256):
        # Check if i is greater than 220.
        if i > 220:
            white_to_black_table.append(0)        
        else:
            white_to_black_table.append(i)
    output_image = cv2.LUT(image, np.array(white_to_black_table).astype("uint8"))
    return output_image

def BlacktoWhite(img):
    black_to_white_table = []

    for i in range(256):
        if i < 50:
            black_to_white_table.append(255)
        else:
            black_to_white_table.append(i)
    output_image = cv2.LUT(image, np.array(black_to_white_table).astype("uint8"))
    return output_image

def applyColorFilter(image, channels_indexes):
    color_table = []
    for i in range(128, 256):
        color_table.extend([i, i])
    output_image = image.copy()
    
    for channel_index in channels_indexes:
        
        output_image[:,:,channel_index] = cv2.LUT(output_image[:,:,channel_index],np.array(color_table).astype("uint8"))
        
    return output_image

prev_gamma = 1.0
intensity_table = []

for i in range(256):
    intensity_table.append(np.clip(a=pow(i/255.0, prev_gamma)*255.0, a_min=0, a_max=255))
    
def changeIntensity(image, scale_factor, channels_indexes):
    
    global prev_gamma, intensity_table
    output_image = image.copy()
     
    gamma = 1.0/scale_factor
    
    if gamma != prev_gamma:
        intensity_table = []
        
        for i in range(256):
            intensity_table.append(np.clip(a=pow(i/255.0, gamma)*255.0, a_min=0, a_max=255))
        
        prev_gamma = gamma
        

    for channel_index in channels_indexes:
        
        output_image[:,:,channel_index] = cv2.LUT(output_image[:,:,channel_index],
                                                  np.array(intensity_table).astype("uint8"))
    
    return output_image

def filter(img , scale_factor , channellist):
    image = applyColorFilter(img, channellist)
    s = changeIntensity(image, scale_factor, channellist)
    return s