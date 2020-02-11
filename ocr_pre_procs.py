#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# 
# ### Image Statistics function

# In[2]:


def im_stats(img):
    (row , col, depth)= img.shape
    print (" Image has",row,"rows,",col,"cols and",depth,"depth")
    
    return (row, col, depth)

def disp_img(img):
    plt.figure(figsize=(18, 18))
    plt.imshow(img)

def img_blur(img, kernel_size= (5,5), rep=1):
    for i in range(rep):
        img = cv2.GaussianBlur(img, kernel_size, 0)
    return img


# ## Display PCBs

# In[3]:


def disp_red_pcb():
    img = cv2.imread("./pcb_im_repo/s_red.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 18))
    plt.imshow(img)
    return img

pcb_red = disp_red_pcb()

def disp_blue_pcb():
    img = cv2.imread("./pcb_im_repo/s_blue.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 18))
    plt.imshow(img)
    return img

pcb_blue = disp_blue_pcb()

def disp_green_pcb():
    img = cv2.imread("./pcb_im_repo/s_green.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 18))
    plt.imshow(img)
    return img

pcb_green = disp_green_pcb()

def disp_black_pcb():
    img = cv2.imread("./pcb_im_repo/s_black.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 18))
    plt.imshow(img)
    return img

pcb_black = disp_black_pcb()

def disp_blue_text_pcb():
    img = cv2.imread("./pcb_im_repo/s_blue_text.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 18))
    plt.imshow(img)
    return img
blue_text_pcb = disp_blue_text_pcb()


# ## Downsampling

# In[4]:


# Scale Factor
scale_fac = 4


# In[5]:


temp_r, temp_c, temp_d = im_stats(pcb_red)
ds_pcb_red = cv2.resize(pcb_red, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_blue)
ds_pcb_blue = cv2.resize(pcb_blue, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_green)
ds_pcb_green = cv2.resize(pcb_green, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_black)
ds_pcb_black = cv2.resize(pcb_black, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(blue_text_pcb)
ds_blue_text_pcb = cv2.resize(blue_text_pcb, ( temp_c//scale_fac, temp_r//scale_fac ))


# In[20]:


temp_r, temp_c, temp_d = im_stats(blue_text_pcb)
ds_blue_text_pcb = cv2.resize(blue_text_pcb, ( temp_c//scale_fac, temp_r//scale_fac ))


# In[ ]:





# In[56]:


temp_mat_black = ds_pcb_black
temp_mat_blue = ds_pcb_blue
temp_mat_red = ds_pcb_red
temp_mat_green = ds_pcb_green

disp_img(temp_mat_black)
disp_img(temp_mat_blue)
disp_img(temp_mat_red)
disp_img(temp_mat_green)

temp_r_black, temp_c_black, temp_d_black = im_stats(temp_mat_black)
temp_r_blue, temp_c_blue, temp_d_blue = im_stats(temp_mat_blue)
temp_r_red, temp_c_red, temp_d_red = im_stats(temp_mat_red)
temp_r_green, temp_c_green, temp_d_green = im_stats(temp_mat_green)

thresh = (0,50,100,150,200,250)

for item in thresh:
    
    for i in range(temp_r_black):
        for j in range(temp_c_black):
            for k in range(temp_d_black):
                if ( (temp_mat_black[i][j][0] > item) and (temp_mat_black[i][j][1] > item) and (temp_mat_black[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_black[i][j] = 0

    for i in range(temp_r_blue):
        for j in range(temp_c_blue):
            for k in range(temp_d_blue):
                if ( (temp_mat_blue[i][j][0] > item) and (temp_mat_blue[i][j][1] > item) and (temp_mat_blue[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_blue[i][j] = 0

    for i in range(temp_r_red):
        for j in range(temp_c_red):
            for k in range(temp_d_red):
                if ( (temp_mat_red[i][j][0] > item) and (temp_mat_red[i][j][1] > item) and (temp_mat_red[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_red[i][j] = 0
    for i in range(temp_r_green):
        for j in range(temp_c_green):
            for k in range(temp_d_green):
                if ( (temp_mat_green[i][j][0] > item) and (temp_mat_green[i][j][1] > item) and (temp_mat_green[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_green[i][j] = 0

    disp_img(temp_mat_black)
    disp_img(temp_mat_blue)
    disp_img(temp_mat_red)
    disp_img(temp_mat_green)
    

    cv2.imwrite( "./saved_images/black_color_thresh_"+str(item)+".jpg", temp_mat_black)
    cv2.imwrite( "./saved_images/blue_color_thresh_"+str(item)+".jpg", temp_mat_blue)
    cv2.imwrite( "./saved_images/red_color_thresh_"+str(item)+".jpg", temp_mat_red)
    cv2.imwrite( "./saved_images/green_color_thresh_"+str(item)+".jpg", temp_mat_green)


# In[21]:


temp_mat_blue_text = ds_blue_text_pcb

disp_img(temp_mat_blue_text)


temp_r_blue_text, temp_c_blue_text, temp_d_blue_text = im_stats(temp_mat_blue_text)

thresh = (0,50,100,150,200,250)

for item in thresh:
    for i in range(temp_r_blue_text):
        for j in range(temp_c_blue_text):
            for k in range(temp_d_blue_text):
                if ( (temp_mat_blue_text[i][j][0] > item) and (temp_mat_blue_text[i][j][1] > item) and (temp_mat_blue_text[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_blue_text[i][j] = 0

    #disp_img(temp_mat_blue_text)
    temp_mat_blue_text = cv2.cvtColor(temp_mat_blue_text, cv2.COLOR_RGB2BGR)

    cv2.imwrite( "./saved_images/blue_text_pcb"+str(item)+".jpg", temp_mat_blue_text)


# In[37]:


temp_mat


# In[ ]:


cv2.imwrite( "./saved_images/color_thresh.jpg", temp_mat)


# ## Gray Threshold

# In[ ]:


temp_mat = cv2.cvtColor(ds_pcb_black, cv2.COLOR_RGB2GRAY)

disp_img(ds_pcb_black)

temp_r, temp_c = temp_mat.shape

for i in range(temp_r):
    for j in range(temp_c):
            if ( temp_mat[i][j] < 200 ):
                temp_mat[i][j] = 0

disp_img(temp_mat)


# ## Histogram

# In[72]:


disp_img(ds_pcb_green)


# In[74]:


img = ds_pcb_green
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# ## Equalized vs Normal Hist Comparision

# In[94]:


img = ds_pcb_black

disp_img(img)

img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

disp_img(img_output)
cv2.imwrite( "./saved_images/orig_black.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR) )
cv2.imwrite( "./saved_images/hist_eq_black.jpg", cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR) )


# In[95]:


img = ds_pcb_black
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[96]:


img = img_output
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[ ]:


temp_img = cv2.imread("./OCR/histogram/black_pcb_hist_eq.png.tif")

temp_enum = (50, 100, 150, 200, 250)

temp_mat = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

temp_r, temp_c, temp_d = temp_mat.shape

for item in temp_enum:
    for i in range(temp_r):
        for j in range(temp_c):
                if ( temp_mat[i][j] < item ):
                    temp_mat[i][j] = 0
                    
    cv2.imwrite( "./saved_images/hist_eq_black_thresh_"+ str(item)+ ".jpg", cv2.cvtColor(temp_mat, cv2.COLOR_RGB2BGR) )
    
disp_img(temp_mat)


# ### Equalized Hist Thresholding

# In[7]:



temp_mat_black = cv2.imread("./OCR/histogram/hist_eq_black.jpg")
temp_mat_blue = cv2.imread("./OCR/histogram/hist_eq_blue.jpg")
temp_mat_red = cv2.imread("./OCR/histogram/hist_eq_red.jpg")
temp_mat_green = cv2.imread("./OCR/histogram/hist_eq_green.jpg")

disp_img(temp_mat_black)
disp_img(temp_mat_blue)
disp_img(temp_mat_red)
disp_img(temp_mat_green)


# In[8]:



temp_mat_black = cv2.imread("./OCR/histogram/hist_eq_black.jpg")
temp_mat_blue = cv2.imread("./OCR/histogram/hist_eq_blue.jpg")
temp_mat_red = cv2.imread("./OCR/histogram/hist_eq_red.jpg")
temp_mat_green = cv2.imread("./OCR/histogram/hist_eq_green.jpg")

disp_img(temp_mat_black)
disp_img(temp_mat_blue)
disp_img(temp_mat_red)
disp_img(temp_mat_green)

temp_r_black, temp_c_black, temp_d_black = im_stats(temp_mat_black)
temp_r_blue, temp_c_blue, temp_d_blue = im_stats(temp_mat_blue)
temp_r_red, temp_c_red, temp_d_red = im_stats(temp_mat_red)
temp_r_green, temp_c_green, temp_d_green = im_stats(temp_mat_green)

thresh = (0,50,100,150,200,250)

for item in thresh:
    
    for i in range(temp_r_black):
        for j in range(temp_c_black):
            for k in range(temp_d_black):
                if ( (temp_mat_black[i][j][0] > item) and (temp_mat_black[i][j][1] > item) and (temp_mat_black[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_black[i][j] = 0

    for i in range(temp_r_blue):
        for j in range(temp_c_blue):
            for k in range(temp_d_blue):
                if ( (temp_mat_blue[i][j][0] > item) and (temp_mat_blue[i][j][1] > item) and (temp_mat_blue[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_blue[i][j] = 0

    for i in range(temp_r_red):
        for j in range(temp_c_red):
            for k in range(temp_d_red):
                if ( (temp_mat_red[i][j][0] > item) and (temp_mat_red[i][j][1] > item) and (temp_mat_red[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_red[i][j] = 0
    for i in range(temp_r_green):
        for j in range(temp_c_green):
            for k in range(temp_d_green):
                if ( (temp_mat_green[i][j][0] > item) and (temp_mat_green[i][j][1] > item) and (temp_mat_green[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_green[i][j] = 0

#     disp_img(temp_mat_black)
#     disp_img(temp_mat_blue)
#     disp_img(temp_mat_red)
#     disp_img(temp_mat_green)

    cv2.imwrite( "./saved_images/black_color_thresh_"+str(item)+".jpg", temp_mat_black)
    cv2.imwrite( "./saved_images/blue_color_thresh_"+str(item)+".jpg", temp_mat_blue)
    cv2.imwrite( "./saved_images/red_color_thresh_"+str(item)+".jpg", temp_mat_red)
    cv2.imwrite( "./saved_images/green_color_thresh_"+str(item)+".jpg", temp_mat_green)


# In[ ]:


temp_mat_blue_text = ds_blue_text_pcb

disp_img(temp_mat_blue_text)


temp_r_blue_text, temp_c_blue_text, temp_d_blue_text = im_stats(temp_mat_blue_text)

thresh = (0,50,100,150,200,250)

for item in thresh:
    

    for i in range(temp_r_blue_text):
        for j in range(temp_c_blue_text):
            for k in range(temp_d_blue_text):
                if ( (temp_mat_blue_text[i][j][0] > item) and (temp_mat_blue_text[i][j][1] > item) and (temp_mat_blue_text[i][j][2] > item) ):
                    continue
                else:
                    temp_mat_blue_text[i][j] = 0

    #disp_img(temp_mat_blue_text)
    temp_mat_blue_text = cv2.cvtColor(temp_mat_blue_text, cv2.COLOR_RGB2BGR )

    cv2.imwrite( "./saved_images/blue_text_pcb"+str(item)+".jpg", temp_mat_blue_text)


# ### Gamma Modification Thresholding

# In[14]:


scale_fac = 4


# In[22]:


temp_r, temp_c, temp_d = im_stats(pcb_red)
ds_pcb_red = cv2.resize(pcb_red, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_blue)
ds_pcb_blue = cv2.resize(pcb_blue, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_green)
ds_pcb_green = cv2.resize(pcb_green, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_black)
ds_pcb_black = cv2.resize(pcb_black, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(blue_text_pcb)
ds_blue_text_pcb = cv2.resize(blue_text_pcb, ( temp_c//scale_fac, temp_r//scale_fac ))


# In[37]:


temp_r, temp_c, temp_d = im_stats(pcb_red)
ds_pcb_red = cv2.resize(pcb_red, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_blue)
ds_pcb_blue = cv2.resize(pcb_blue, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_green)
ds_pcb_green = cv2.resize(pcb_green, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(pcb_black)
ds_pcb_black = cv2.resize(pcb_black, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_r, temp_c, temp_d = im_stats(blue_text_pcb)
ds_blue_text_pcb = cv2.resize(blue_text_pcb, ( temp_c//scale_fac, temp_r//scale_fac ))

temp_mat_black = ds_pcb_black
temp_mat_blue = ds_pcb_blue
temp_mat_red = ds_pcb_red
temp_mat_green = ds_pcb_green

disp_img(temp_mat_black)
disp_img(temp_mat_blue)
disp_img(temp_mat_red)
disp_img(temp_mat_green)

temp_r_black, temp_c_black, temp_d_black = im_stats(temp_mat_black)
temp_r_blue, temp_c_blue, temp_d_blue = im_stats(temp_mat_blue)
temp_r_red, temp_c_red, temp_d_red = im_stats(temp_mat_red)
temp_r_green, temp_c_green, temp_d_green = im_stats(temp_mat_green)

#thresh = (50,100,150,200,250)
thresh = (50,100,127,200)

### Non linearirty is based on a exponential scale based on (value/thresh)

for item in thresh:
    
    print("Now Evaluating Threshold:"+str(item))
    
    temp_r, temp_c, temp_d = im_stats(pcb_black)
    ds_pcb_black = cv2.resize(pcb_black, ( temp_c//scale_fac, temp_r//scale_fac ))
    temp_mat_black = ds_pcb_black
    temp_r_black, temp_c_black, temp_d_black = im_stats(temp_mat_black)
    
    temp_r, temp_c, temp_d = im_stats(pcb_blue)
    ds_pcb_blue = cv2.resize(pcb_blue, ( temp_c//scale_fac, temp_r//scale_fac ))
    temp_mat_blue = ds_pcb_blue
    temp_r_blue, temp_c_blue, temp_d_blue = im_stats(temp_mat_blue)
    
    temp_r, temp_c, temp_d = im_stats(pcb_red)
    ds_pcb_red = cv2.resize(pcb_red, ( temp_c//scale_fac, temp_r//scale_fac ))
    temp_mat_red = ds_pcb_red
    temp_r_red, temp_c_red, temp_d_red = im_stats(temp_mat_red)
    
    temp_r, temp_c, temp_d = im_stats(pcb_green)
    ds_pcb_green = cv2.resize(pcb_green, ( temp_c//scale_fac, temp_r//scale_fac ))
    temp_mat_green = ds_pcb_green
    temp_r_green, temp_c_green, temp_d_green = im_stats(temp_mat_green)
    
    for i in range(temp_r_black):
        for j in range(temp_c_black):
            for k in range(temp_d_black):
                
                if (temp_mat_black[i][j][0] > item and temp_mat_black[i][j][1] > item and temp_mat_black[i][j][2] > item):
                    
                    temp_mat_black[i][j][k] = 255 if (temp_mat_black[i][j][k] ** (temp_mat_black[i][j][k]/item))>255 else (temp_mat_black[i][j][k] ** (temp_mat_black[i][j][k]/item))
                
                else:
                    temp_mat_black[i][j][k] = 0
    
    
    for i in range(temp_r_blue):
        for j in range(temp_c_blue):
            for k in range(temp_d_blue):
                
                if (temp_mat_blue[i][j][0] > item and temp_mat_blue[i][j][1] > item and temp_mat_blue[i][j][2] > item):
                    
                    temp_mat_blue[i][j][k] = 255 if (temp_mat_blue[i][j][k] ** (temp_mat_blue[i][j][k]/item))>255 else (temp_mat_blue[i][j][k] ** (temp_mat_blue[i][j][k]/item))
                
                else:
                    temp_mat_blue[i][j][k] = 0

                    
    for i in range(temp_r_red):
        for j in range(temp_c_red):
            for k in range(temp_d_red):
                
                if (temp_mat_red[i][j][0] > item and temp_mat_red[i][j][1] > item and temp_mat_red[i][j][2] > item):
                    
                    temp_mat_red[i][j][k] = 255 if (temp_mat_red[i][j][k] ** (temp_mat_red[i][j][k]/item))>255 else (temp_mat_red[i][j][k] ** (temp_mat_red[i][j][k]/item))
                
                else:
                    temp_mat_red[i][j][k] = 0

    for i in range(temp_r_green):
        for j in range(temp_c_green):
            for k in range(temp_d_green):
                
                if (temp_mat_green[i][j][0] > item and temp_mat_green[i][j][1] > item and temp_mat_green[i][j][2] > item):
                    
                    temp_mat_green[i][j][k] = 255 if (temp_mat_green[i][j][k] ** (temp_mat_green[i][j][k]/item))>255 else (temp_mat_green[i][j][k] ** (temp_mat_green[i][j][k]/item))
                
                else:
                    temp_mat_green[i][j][k] = 0

    #disp_img(temp_mat_black)
    cv2.imwrite( "./saved_images/black_pcb_contrast_stretch"+str(item)+".jpg", cv2.cvtColor(temp_mat_black, cv2.COLOR_RGB2BGR) )
    cv2.imwrite( "./saved_images/blue_pcb_contrast_stretch"+str(item)+".jpg", cv2.cvtColor(temp_mat_blue, cv2.COLOR_RGB2BGR) )
    cv2.imwrite( "./saved_images/red_pcb_contrast_stretch"+str(item)+".jpg", cv2.cvtColor(temp_mat_red, cv2.COLOR_RGB2BGR) )
    cv2.imwrite( "./saved_images/green_pcb_contrast_stretch"+str(item)+".jpg", cv2.cvtColor(temp_mat_green, cv2.COLOR_RGB2BGR) )
    


# ### Histogram Gen

# In[ ]:


img = cv2.imread("./pcb_im_repo/s_red.tif")
    
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[ ]:





# In[36]:


img = cv2.imread('./OCR/ocr_text_thresh/contrast_stretch/black_pcb_contrast_stretch50.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()


# In[ ]:




