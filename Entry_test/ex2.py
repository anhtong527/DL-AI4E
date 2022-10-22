# Entry test phần lập trình: Bài 2
import cv2

# read the image
image = cv2.imread('image.png')

#print(image.shape)
height, width, channel = image.shape

# test showing base image
cv2.imshow("Base image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def crop_image(img, cropped_height, cropped_width):
    """
    img: base image read by cv2.imread()
    cropped_height: new height (int type)
    cropped_width: new width (int type)

    Return: the image that is cropped
    """

    cropped_image = img[0:cropped_height, 0:cropped_width]

    cv2.imshow('Cropped image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_image

def resize_image(img, resized_height, resized_width):
    """
    Resize by lowering the image pixels
    
    img: base image read by cv2.imread()
    resized_height: new height (int type)    
    resized_width: new width (int type)
    
    Return image that is resized
    """

    resized_image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    # INTER_AREA interpolation method: resampling using pixel area relation

    cv2.imshow('resized image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized_image

def gaussian_blurring(img, kernel_size):
    """
    Smoothing the image by applying Gaussian blur method

    img: base iamge read by cv2.imread()
    kernel_size: a tuple (x,y); x,y are positive and odd; higher x,y -> more blurred
    
    Return: the image that is smoothed
    """

    blurred_image = cv2.GaussianBlur(img, kernel_size, cv2.BORDER_DEFAULT)
    # cv2.BORDER_DEFAULT: type of border (use default)

    cv2.imshow('Blurred image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blurred_image

def detect_edges_sobel(img):
    """
    Detect edges by using Sobel method
    
    img: base image read by cv2.imread()
    
    Return: three types of Sobel method [X, Y, XY]
    """
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), cv2.BORDER_DEFAULT)

    # Sobel Edge Detection

    # Sobel Edge Detection on the X axis
    sobelx_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    # ddepth specifies the precision of the output image
    # dx and dy specify the order of the derivative in each direction

    # Sobel Edge Detection on the Y axis
    sobely_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

    # Combined X and Y Sobel Edge Detection
    sobelxy_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    cv2.imshow('Sobel X', sobelx_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Sobel Y', sobely_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Sobel XY', sobelxy_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sobelx_img, sobely_img, sobelxy_img

def detect_edges_canny(img, threshold1, threshold2):
    """
    Detect edges by using Canny method
    
    img: base image read by cv2.imread()
    
    Return: Canny Edge Detection Image
    """
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), cv2.BORDER_DEFAULT)

    canny_edges_img = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    # gradient magnitude value of pixel < threshold1 => be EXCLUDED in the final edge map
    # gradient magnitude value of pixel > threshold2 => be INCLUDED in the final edge map

    cv2.imshow('Canny edges', canny_edges_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return canny_edges_img


# 1. Crop the 1/4 top left corner of the image
# Do it by slicing the image
cropped_height = int(height/2)
cropped_width = int(width/2)

crop_image(image, cropped_height, cropped_width)


# 2. Resize half height, width of the image
resized_height = int(height/2)
resized_width = int(width/2)

resize_image(image, resized_height, resized_width)


# 3. Gaussian Blurring
kernel_size = (7,7)

blurred_image = gaussian_blurring(image, kernel_size)


# 4. detect "edges" in the image
detect_edges_sobel(image)

detect_edges_canny(image, 5, 15)