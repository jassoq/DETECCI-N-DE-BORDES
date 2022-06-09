import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import FN
import  cv2
import math

################################ Robert Sobel Prewitt ###############################
if __name__ == '__main__':


    gray_imager = np.asarray(Image.open(r'IMG3.jpg').convert('L'))

    robert = [[[0, 1], [-1, 0]], [[1, 0], [0, -1]]]
    
    sobel = [[[1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]

    prewitt = [[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[-1, -1, -1], [0, 0, 0], [1,1 , 1]]]

    robert_img = FN.suanzi(gray_imager, robert, 1)
    sobel_img = FN.suanzi(gray_imager, sobel, 1)
    prewitt_img = FN.suanzi(gray_imager, prewitt, 1)


    fig = plt.figure()
    ax1 = fig.add_subplot(231)

    ax1.set_title('EDG')
    ax1.imshow(gray_imager, cmap='gray', vmin=0, vmax=255)

    ax1 = fig.add_subplot(234)
    ax1.set_title('ROBERT')
    ax1.imshow(robert_img, cmap='gray', vmin=0, vmax=255)

    ax1 = fig.add_subplot(235)
    ax1.set_title('SOBEL')
    ax1.imshow(sobel_img, cmap='gray', vmin=0, vmax=255)

    ax1 = fig.add_subplot(236)
    ax1.set_title('PREWITT')
    ax1.imshow(prewitt_img, cmap='gray', vmin=0, vmax=255)


    plt.show()

################################ Isotropico ###############################

img=cv2.imread("IMG3.jpg",0)   
img=np.float32(img)
height, width = img.shape[0:2]

IsotropicoX=[[0 for i in range(width)]for j in range(height)]
IsotropicoY=[[0 for i in range(width)]for j in range(height)]

for i in range(height-1):
    for j in range(width-1):
        IsotropicoX[i][j]=-img[i-1,j-1]+img[i+1,j-1]-(np.sqrt(2)*img[i-1,j])+np.sqrt(2)*img[i+1,j]-img[i-1,j+1]+img[i+1,j+1]
        IsotropicoY[i][j]=-img[i-1,j-1]-(np.sqrt(2)*(img[i,j-1]))-img[i+1,j-1]+img[i-1,j+1]+(np.sqrt(2)*(img[i,j+1]))+img[i+1,j+1]
        if(IsotropicoY[i][j]<0):
            IsotropicoY[i][j]=0
        if(IsotropicoX[i][j]<0):
            IsotropicoX[i][j]=0

IsotropicoX=np.uint8(IsotropicoX)
IsotropicoY=np.uint8(IsotropicoY)
IsotropicoXY = cv2.add(IsotropicoX,IsotropicoY)

cv2.imshow("Isotropico X", IsotropicoX)
cv2.imshow("Isotropico Y", IsotropicoY)
cv2.imshow("Isotropico",IsotropicoXY)
cv2.waitKey(0)


################################ KIRSCH ###############################


KIRSCH_0     = np.array([[ -3,-3, 5] , [-3,0, 5] , [-3,-3, 5]], dtype=np.float32) 
KIRSCH_45    = np.array([[ -3, 5, 5] , [-3,0, 5] , [-3,-3,-3]], dtype=np.float32) 
KIRSCH_90    = np.array([[  5, 5, 5] , [-3,0,-3] , [-3,-3,-3]], dtype=np.float32) 
KIRSCH_135   = np.array([[  5, 5,-3] , [ 5,0,-3] , [-3,-3,-3]], dtype=np.float32) 
KIRSCH_180   = np.array([[  5,-3,-3] , [ 5,0,-3] , [ 5,-3,-3]], dtype=np.float32) 
KIRSCH_225   = np.array([[ -3,-3,-3] , [ 5,0,-3] , [ 5, 5,-3]], dtype=np.float32) 
KIRSCH_270   = np.array([[ -3,-3,-3] , [-3,0,-3] , [ 5, 5, 5]], dtype=np.float32) 
KIRSCH_315   = np.array([[ -3,-3,-3] , [-3,0, 5] , [-3, 5, 5]], dtype=np.float32) 

def kirsch_filter(img) :
    """ Return a gray-scale image that's been Kirsch edge filtered. """
    if  img.ndim > 2 :
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fimg    = np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_0),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_45),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_90),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_135),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_180),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_225),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, KIRSCH_270),
                            cv2.filter2D(img, cv2.CV_8U, KIRSCH_315),
                           )))))))
    return(fimg)


def threshold(img, sig = None) :
    """ Threshold a gray image in a way that usually makes sense. """
    if  img.ndim > 2 :
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    med     = np.median(img)
    if  sig is None :
        sig = 0.0       # note: sig can be negative. Another way: Use the %'th percentile-ish pixel.
    co      = int(min(255, max(0, (1.0 + sig) * med)))
    return(cv2.threshold(img, co, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1])



if  __name__ == '__main__' :
    import  os
    import  sys

    fn      = "IMG4.jpg"
    if  len(sys.argv) > 1 :
        fn  = sys.argv.pop(1)
    sig     = None
    if  len(sys.argv) > 1 :
        sig = float(sys.argv.pop(1))

    img     = cv2.imread(fn)

    kimg    = kirsch_filter(img)    # make each pixel the maximum edginess value
    tkimg    = threshold(kimg, sig)  # make the edges stand out
    tkimg    = 255 - tkimg            # invert the image to make the edges white

################################ Robinson ###############################

ROB_0     = np.array([[-1, 0, 1] , [-2,0, 2] , [-1, 0, 1]], dtype=np.float32) 
ROB_45    = np.array([[ 0, 1, 2] , [-1,0, 1] , [-2,-1, 0]], dtype=np.float32) 
ROB_90    = np.array([[ 1, 2, 1] , [ 0,0, 0] , [-1,-2,-1]], dtype=np.float32) 
ROB_135   = np.array([[ 2, 1, 0] , [ 1,0,-1] , [ 0,-1,-2]], dtype=np.float32) 
ROB_180   = np.array([[ 1, 0,-1] , [ 2,0,-2] , [ 1, 0,-1]], dtype=np.float32) 
ROB_225   = np.array([[ 0,-1,-2] , [ 1,0,-1] , [ 2, 1, 0]], dtype=np.float32) 
ROB_270   = np.array([[-1,-2,-1] , [ 0,0, 0] , [ 1, 2, 1]], dtype=np.float32) 
ROB_315   = np.array([[-2,-1, 0] , [-1,0, 1] , [ 0, 1, 2]], dtype=np.float32) 

def robinson_filter(img) :
    """ Return a gray-scale image that's been Kirsch edge filtered. """
    if  img.ndim > 2 :
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fimg    = np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_0),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_45),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_90),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_135),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_180),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_225),
              np.maximum(cv2.filter2D(img, cv2.CV_8U, ROB_270),
                            cv2.filter2D(img, cv2.CV_8U, ROB_315),)))))))
    return(fimg)


def threshold(img, sig = None) :
    """ Threshold a gray image in a way that usually makes sense. """
    if  img.ndim > 2 :
        img = cv2.cvtColor(img)
    med     = np.median(img)
    if  sig is None :
        sig = 0.0       # note: sig can be negative. Another way: Use the %'th percentile-ish pixel.
    co      = int(min(255, max(0, (1.0 + sig) * med)))
    return(cv2.threshold(img, co, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1])



if  __name__ == '__main__' :
    import  os
    import  sys

    fn      = "IMG4.jpg"
    if  len(sys.argv) > 1 :
        fn  = sys.argv.pop(1)
    sig     = None
    if  len(sys.argv) > 1 :
        sig = float(sys.argv.pop(1))

    img     = cv2.imread(fn)

    rimg    = robinson_filter(img)    # make each pixel the maximum edginess value
    trimg    = threshold(rimg, sig)  # make the edges stand out
    trimg    = 255 - trimg            # invert the image to make the edges white


    cv2.imshow('Robinson',trimg)
    cv2.imshow('Kirsh', tkimg)
    cv2.imshow('Original',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()






