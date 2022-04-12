from PIL import Image
from matplotlib import image
import streamlit as st
from streamlit_ace import st_ace
import cv2 as cv
import numpy as np
import pixelfile as pf

st.title("Pixel Editor")

a = 23
p = st.sidebar.file_uploader('sd')
if(p):
    i = cv.imread(p.name)
    s = cv.resize(i , (500 , 500))

    a,b,c = s.shape

    for k in range(a):
        for g in range(b):
            for h in range(c):
                try:
                    s[k,g,h] = 144
                except:
                    pass


    cv.imshow('sd' , s)

    cv.waitKey(0)

#color filter by channel
'''

import cv2 as cv

def filter(img):
    s = img
    w , h , c = img.shape
    
    for k in range(int(w)):
        for z in range(int(h)):
            s[k,z,2] = 34

    return s

'''


'''
    import cv2 as cv
    
    def filter(img):
        w,h,c = img.shape
        res = img
        for a in range(w):
            for b in range(h):
                for z in range(c):
                    try:
                        res[a , b , z] = (img[-a,b,z])
                    except:
                        pass
        return res

'''

'''
    import cv2 as cv
    import numpy as np 
    import math

    def filter(img):
        w,h,c = img.shape
        res = img
        for a in range(w):
            for b in range(h):
                for z in range(c):
                    try:
                        res[a , b , z] = 255 - img[-a,b,z]
                    except:
                        pass
        return res
    
    '''



