import cv2 as cv

def filter(img):
    s = img
    w , h , c = img.shape
    
    for k in range(int(w)):
        for z in range(int(h)):
            s[k,z,0] = 94

    return s