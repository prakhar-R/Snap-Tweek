from cv2 import FONT_HERSHEY_DUPLEX
import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageColor
import torch 
import pixelfile as pf
from streamlit_ace import st_ace
from torchvision import transforms , models 
from PIL import Image 
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import colorfilters as cf
from scipy import ndimage

#---------------------------transformations--------------------------------------------------------------------------------------

def wrapaffine(img):
    #image = array
    row , column , ch = img.shape
    pt1 = np.float32([[50,50] , [200 , 50] , [50,200]])
    pt2 = np.float32([[50,100] , [200 , 50] , [150,200]])
    pts = cv.getAffineTransform(pt1 , pt2)
    result = cv.warpAffine(img , pts , (column , row))
    res_conv = Image.fromarray(result)
    return res_conv

def rotation(img , theta):
    rotated = ndimage.rotate(img , theta)
    result = Image.fromarray(rotated)
    return result

def flipping(img , theta):
    flipped_image = cv.flip(img , theta)
    flip = Image.fromarray(flipped_image)
    return flip

#---------------------------transformations--------------------------------------------------------------------------------------

#-------------------------------------------------Brush--------------------------------------------------------------------------

def Text(img):

    img = np.array(Image.open(input_image[-1]))
    txt_res = st.sidebar.text_input('Text to Add')
    sl_x = st.sidebar.slider('x coor', 1 , img.shape[1], step=1)
    sl_y = st.sidebar.slider('y coor', 1 , img.shape[0], step=1)
    size = st.sidebar.slider('Size',1 , 20 , 11, step=1)
    txt_thick = st.sidebar.slider('thickness', 1 , 20 , 13,  step=1)
    
    red = st.sidebar.text_input('Red' , 0)
    green = st.sidebar.text_input('Green' , 0)
    blue = st.sidebar.text_input('Blue' , 0)
    
    if(txt_res != ""):
        i = cv.putText(img , txt_res , (sl_x , sl_y) , cv.FONT_HERSHEY_COMPLEX , size , (int(red),int(green),int(blue)) , txt_thick)
        input_image.append(i)

def Draw(img):
    
        drawing_mode = st.sidebar.selectbox(    
            "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
        )

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

            

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(img) if bg_image else None,
            update_streamlit=True,
            height=img.shape[0],
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )
    
#-------------------------------------------------Brush--------------------------------------------------------------------------

#-------------------------------------------------------filters------------------------------------------------------------------ 
 
def sketch(img):
    
    def Dodge(img , mask):
        return cv.divide(img , 255-mask , scale = 256)
    
    curr_img = img
    gray_img = cv.cvtColor(curr_img , cv.COLOR_BGRA2GRAY)
    slider = st.sidebar.slider('intensity', 5, 81, 33, step=2)
    invert_img = cv.bitwise_not(gray_img)
    blurring = cv.GaussianBlur(invert_img , (slider,slider) , sigmaX=0 , sigmaY=0)
    img_blending = Dodge(gray_img , blurring)
    return img_blending

def Gray_Filter(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def blackandWhite(img):
    converted_img = img
    gray_scale = cv.cvtColor(converted_img, cv.COLOR_RGB2GRAY)
    slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
    (thresh, blackAndWhiteImage) = cv.threshold(gray_scale, slider, 255, cv.THRESH_BINARY)

def BlurEffect(img):
    converted_img = img
    slider = st.sidebar.slider('intensity', 5, 81, 33, step=2)
    converted_img = cv.cvtColor(converted_img, cv.COLOR_RGB2BGR)
    blur_image = cv.GaussianBlur(converted_img, (slider,slider), 0, 0)
    
def BW_filter(img , slider):
    gray_scale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    slider = st.sidebar.slider('intensity', 1, 255, 127, step=1)
    (thresh, blackAndWhiteImage) = cv.threshold(gray_scale, slider, 255, cv.THRESH_BINARY)
    return blackAndWhiteImage

def Water_Art_Filter(img):
    #1)impurities
    imgs = img
    img_clear = cv.medianBlur( imgs,  3)
    img_clear = cv.medianBlur(img_clear , 3)
    img_clear = cv.medianBlur(img_clear , 3)
    
    #2) mixing of colrs propotional to sigma
    img_clear = cv.edgePreservingFilter(img_clear , sigma_s=5)
    
    #3) image filterring (blurring with intensity)
    img_filter = cv.bilateralFilter(img_clear , 3, 10 , 5)
    img_filter = cv.bilateralFilter(img_filter , 3, 20 , 10)
    img_filter = cv.bilateralFilter(img_filter , 3, 20 , 10)
    img_filter = cv.bilateralFilter(img_filter , 3, 30 , 10)
    img_filter = cv.bilateralFilter(img_filter , 3, 30 , 10)
    img_filter = cv.bilateralFilter(img_filter , 3, 30 , 10)
    
    #4)Blur -> sharpining
    gauss_mask = cv.GaussianBlur(img_filter , (7,7) , 2)
    img_sharp = cv.addWeighted(img_filter , 1.5 , gauss_mask , -0.5 , 0)
    img_sharp = cv.addWeighted(img_sharp , 1.4 , gauss_mask , -0.2 , 0)
    
    return img_sharp

def Style_transfer(content_image , style_images , iterations = 1000):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19(pretrained=True).features
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    def model_activations(input,model):
        layers = {
        '0' : 'conv1_1',
        '5' : 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
        }
        features = {}
        x = input
        x = x.unsqueeze(0) # 2x2 - > 1x1
        for name,layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x 
        
        return features


    transform = transforms.Compose([transforms.Resize(300),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    content =Image.open(content_image).convert("RGB")
    content = transform(content).to(device)
    print("COntent shape => ", content.shape)
    style = Image.open(style_images).convert("RGB")
    style = transform(style).to(device)
    def imcnvt(image):
        x = image.to("cpu").clone().detach().numpy().squeeze()
        x = x.transpose(1,2,0)
        x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
        return np.clip(x,0,1)


    def gram_matrix(imgfeature):
        _,d,h,w = imgfeature.size()
        imgfeature = imgfeature.view(d,h*w)
        gram_mat = torch.mm(imgfeature,imgfeature.t())
        
        return gram_mat


    target = content.clone().requires_grad_(True).to(device)

    #set device to cuda if available
    #print("device = ",device)


    style_features = model_activations(style,model)
    content_features = model_activations(content,model)

    style_wt_meas = {"conv1_1" : 1.0, 
                    "conv2_1" : 0.8,
                    "conv3_1" : 0.4,
                    "conv4_1" : 0.2,
                    "conv5_1" : 0.1}

    style_grams = {layer:gram_matrix(style_features[layer]) for layer in style_features}

    content_wt = 100
    style_wt = 1e8

    print_after = iterations
    epochs = iterations
    optimizer = torch.optim.Adam([target],lr=0.007)

    for i in range(1,epochs+1):
        target_features = model_activations(target,model)
        content_loss = torch.mean((content_features['conv4_2']-target_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_wt_meas:
            style_gram = style_grams[layer]
            target_gram = target_features[layer]
            _,d,w,h = target_gram.shape
            target_gram = gram_matrix(target_gram)

            style_loss += (style_wt_meas[layer]*torch.mean((target_gram-style_gram)**2))/d*w*h
        
        total_loss = content_wt*content_loss + style_wt*style_loss 
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i%print_after == 0:
            return (imcnvt(target))
       
#-------------------------------------------------------filters------------------------------------------------------------------ 

def PxelEditor(img):
    code = st_ace(language='python' , theme = 'ambiance') 

    f  = open("pixelfile.py" , 'w')

    if(code):
        
        f.writelines(code)
        f.close()
    
    
    Applybutton = st.sidebar.button('Apply')
    
    st.sidebar.write('See option main.py for boiler plate code')
    
    if (Applybutton):
        import pixelfile as pf
        try:
            use_photo = cv.imread(photo.name)
            #s = cv.resize(use_photo , (500,500))
            s = use_photo
            img_result = pf.filter(s)
            cv.imshow('Pixel Editor' , img_result)
            cv.waitKey(0)
        except:
            pass

#---------------------------------------------------OUTLINE----------------------------------------------------------------------
st.title("Snap~Tweak")

photo = st.sidebar.file_uploader("Choose Image" , type = ['png' , 'jpeg' , 'png' , 'jpg'])

input_image = []

#cache later
if(photo):
    
    input_image.append(photo)    

    optionList = st.sidebar.radio("select one" , ['None' , 'Transformations' , 'Filters' , 'Text' , 'Draw' ,"Pixel Editor"])
    
    if(optionList == "Transformations"):
        Transformations = st.sidebar.selectbox ("Transformations" , ["none" , "Wrapaffine" , "Rotation" , "Flip"])
    
        if(Transformations == "Wrapaffine"):
            res = wrapaffine(np.array(Image.open(input_image[-1])))
            input_image.append(res)
        
        if(Transformations == "Rotation"):
            slider = st.sidebar.slider('rotation',-180, 180, 0, step=1)
            res_rot = rotation(np.array(Image.open(input_image[-1])),int(slider))
            input_image.append(res_rot)
        
        if(Transformations == "Flip"):
            slider = st.sidebar.slider('rotation',-1, 1, -1, step=2)
            res_flip = flipping(np.array(Image.open(input_image[-1])) ,slider)
            input_image.append(res_flip)
    
    if(optionList == "Filters"):
        filter_option = st.sidebar.selectbox("Filters" , ["none","sketch" , "Color" , "Blur" , "Water Art" , "Style Transfer"])
    
        if (filter_option == "Color"):
            value = st.sidebar.radio("choose" , ["Gray" , "Black&White" , 'custom'])
            if(value == "Gray"):
                img_gray = Gray_Filter(np.array(Image.open(input_image[-1])))
                input_image.append(img_gray)
            
            if(value == "Black&White"):
                gray_scale = cv.cvtColor(np.array(Image.open(input_image[-1])), cv.COLOR_RGB2GRAY)
                slider = st.sidebar.slider('intensity', 1, 255, 127, step=1)
                (thresh, blackAndWhiteImage) = cv.threshold(gray_scale, slider, 255, cv.THRESH_BINARY)
                input_image.append(blackAndWhiteImage)
            
            if(value == "custom"):
                scalingFactor = st.sidebar.slider('scaler' , 1 , 100 , step = 1)
                st.write("Channels")
                r = st.sidebar.checkbox('Red')
                g = st.sidebar.checkbox('Green')
                b = st.sidebar.checkbox('Blue')
                channels = []
                if(r):
                   channels = [0]
                if(r and g ):
                    channels = [0,1]
                if(r and b):
                    channels = [0,2]
                if(r and g and b):
                    channels = [0,1,2]
                if(g and b):
                    channels = [1,2]
                if(g):
                    channels = [1]
                if(b):
                    channels = [2]
                m = cf.filter(cv.imread(input_image[-1].name), scalingFactor/100 , channels) 
                
                input_image.append(m)
        
        if(filter_option == "sketch"):
            
            def Dodge(img , mask):
                return cv.divide(img , 255-mask , scale = 256)
            
            curr_img = np.array(Image.open(input_image[-1]))
            gray_img = cv.cvtColor(curr_img , cv.COLOR_BGRA2GRAY)
            slider1 = st.sidebar.slider('intensity', 5, 81, 33, step=2)
            invert_img = cv.bitwise_not(gray_img)
            blurring = cv.GaussianBlur(invert_img , (slider1,slider1) , sigmaX=0 , sigmaY=0)
            img_blending = Dodge(gray_img , blurring)
            sketched = img_blending
            input_image.append(sketched)
                
        if(filter_option == "Blur"):   
            img = np.array(Image.open(input_image[-1]))
            slider2 = st.sidebar.slider('intensity', 5, 81, 33, step=2)
            #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            blur_image = cv.GaussianBlur(img, (slider2,slider2), 0, 0)  
            input_image.append(blur_image)
        
        if (filter_option == "Water Art"):
            
            imgs = np.array(Image.open(input_image[-1]))
            img_clear = cv.medianBlur( imgs,  3)
            img_clear = cv.medianBlur(img_clear , 3)
            img_clear = cv.medianBlur(img_clear , 3)
            
            img_clear = cv.edgePreservingFilter(img_clear , sigma_s=5)
            
            img_filter = cv.bilateralFilter(img_clear , 3, 10 , 5)
            img_filter = cv.bilateralFilter(img_filter , 3, 20 , 10)
            img_filter = cv.bilateralFilter(img_filter , 3, 20 , 10)
            img_filter = cv.bilateralFilter(img_filter , 3, 30 , 10)
            img_filter = cv.bilateralFilter(img_filter , 3, 30 , 10)
            img_filter = cv.bilateralFilter(img_filter , 3, 30 , 10)
            
            slider3 = st.sidebar.slider('intensity', 5, 81, 33, step=2)
            gauss_mask = cv.GaussianBlur(img_filter , (slider3,slider3) , 2)
            img_sharp = cv.addWeighted(img_filter , 1.5 , gauss_mask , -0.5 , 0)
            img_sharp = cv.addWeighted(img_sharp , 1.4 , gauss_mask , -0.2 , 0)
            
            input_image.append(img_sharp)

        if (filter_option == "Style Transfer"):
            style = st.sidebar.file_uploader("styling image" , type = ['png' , 'jpeg' , 'png' , 'jpg'])
            if(style):
                pressed = st.sidebar.button("start")
                if(pressed):
                    edited_photo = Style_transfer(input_image[-1] , style , 100)
                    input_image.append(edited_photo)
            
    if(optionList == "Text"):
        img = np.array(Image.open(input_image[-1]))
        
        txt_res = st.sidebar.text_area('Text to Add')
        sl_x = st.sidebar.slider('x coor', 1 , img.shape[1], step=1)
        sl_y = st.sidebar.slider('y coor', 1 , img.shape[0], step=1)
        size = st.sidebar.slider('Size',1 , 20 , 11, step=1)
        txt_thick = st.sidebar.slider('thickness', 1 , 20 , 13,  step=1)
        print(txt_res)
        
        font_array = [cv.FONT_HERSHEY_COMPLEX_SMALL , cv.FONT_HERSHEY_DUPLEX , cv.FONT_HERSHEY_DUPLEX , cv.FONT_HERSHEY_PLAIN , cv.FONT_HERSHEY_SCRIPT_SIMPLEX , cv.FONT_HERSHEY_SIMPLEX , cv.FONT_ITALIC , cv.FONT_HERSHEY_PLAIN , cv.FONT_HERSHEY_SIMPLEX]
        
        num_font = []
        
        for s in range(len(font_array)):
            num_font.append(s)
        
        font = st.sidebar.selectbox("Font", num_font)
        
        color = st.sidebar.color_picker('text color')
        
        if(txt_res != ""):
            i = cv.putText(img , txt_res , (sl_x , sl_y) , font_array[font] , size , ImageColor.getcolor(color, "RGB") , txt_thick)
            input_image.append(i)

if(photo):
    for i in input_image:
        st.image(i)

    if(optionList == "Draw"):
            
            img = input_image[-1]
            
            drawing_mode = st.sidebar.selectbox(  
                "drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
            )

            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
            if drawing_mode == 'point':
                point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
            stroke_color = st.sidebar.color_picker("Stroke color hex: ")
            bg_image = Image.open(img) 

            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                #background_color=bg_color,
                background_image=bg_image,
                update_streamlit=True,
                height=550,
                width=700,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="canvas",
            )

    if(optionList == "Pixel Editor"):
        code = st_ace(language='python' , theme = 'ambiance') 

        f  = open("pixelfile.py" , 'w')

        if(code):
           
            f.writelines(code)
            f.close()
        
        img2 = st.sidebar.file_uploader('transition')
        
        Applybutton = st.sidebar.button('Apply')
        
        st.sidebar.write('See option PixelEditor.py for boiler plate code')
        
        if (Applybutton):
            import pixelfile as pf
            try:
                use_photo = cv.imread(photo.name)
                #s = cv.resize(use_photo , (500,500))
                s = use_photo
                img_result = pf.filter(cv.resize(s,(500 , 400)))
                cv.imshow('Pixel Editor' , img_result)
                cv.waitKey(0)
                cv.destroyAllWindows()
            except:
                pass
            
    
    
        