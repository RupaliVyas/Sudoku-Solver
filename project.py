import cv2
import numpy as np,sys
import matplotlib.pyplot as plt
import operator
from scipy.spatial import distance
import tensorflow as tf
from tensorflow import keras


C = cv2.cv2.imread("images/new_img.png",0) # Read image in gray scale


def pre_process_image(img, skip_dilate=False):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    
    proc = cv2.cv2.GaussianBlur(img.copy(), (11,11), 0)

    proc = cv2.cv2.adaptiveThreshold(proc, 255, cv2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.cv2.THRESH_BINARY, 13, 2)

   
    proc = cv2.cv2.bitwise_not(proc)

    if not skip_dilate:
      
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.cv2.dilate(proc, kernel)

    plt.imshow(proc, cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()
    return proc




def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""
    _, contours,h = cv2.cv2.findContours(img.copy(), cv2.cv2.RETR_EXTERNAL, cv2.cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.cv2.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]

   
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key= operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key= operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key= operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key= operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]



processed_img = pre_process_image(C, skip_dilate=False)

def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""


    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    
    side = max([
        distance.euclidean(bottom_right, top_right),
        distance.euclidean(top_left, bottom_left),
        distance.euclidean(bottom_right, bottom_left),
        distance.euclidean(top_left, top_right)
        ])

    height= distance.euclidean(bottom_right, top_right)
    width=  distance.euclidean(bottom_right, bottom_left)

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    m = cv2.cv2.getPerspectiveTransform(src, dst)

    warp = cv2.cv2.warpPerspective(img, m, (int(side), int(side)))
    plt.imshow(warp, cmap='gray')
    plt.title('Warped Image')
    plt.show()
    return warp,side,height, width




#SEPARATES NUMBERS IN THE GRID

wrped,side,hei,wid = crop_and_warp(processed_img,find_corners_of_largest_polygon(processed_img))


#hei=941.6666076696147
#wid=978.0250508039147
#no= wrped[0:int(104.62962307440164),int(217.33890017864772):int(326.0083502679716) ]
#cv2.cv2.imshow("cropped", no)

pts=side/9
pts2=side/9

rows = []
col=[]
hei_ptr=0
wid_ptr=0
for m in range(10):
    rows.append(hei_ptr)
    hei_ptr+=pts
for j in range(10):
    col.append(wid_ptr)
    wid_ptr+=pts2



#print(rows[:-1])
#print(col[:-1])

nos=[]        #array of numbers
for row in rows[:-1]:
    for c in col[:-1]:
        cropped_new= wrped[int(row):int(row+pts), int(c):int(c+pts2)]
        nos.append(cropped_new)


# AFTER RUNNING THE CODE PRESS KEYS TO CYCLE THROUGH



plt.imshow(nos[0], cmap='gray')
plt.title('Extracted Cell')
plt.show()


def prepare(no):
    
    IMG_SIZE = 28
    img_array = no.copy()
    new_img= img_array[7:(img_array.shape[0]-5), 7:(img_array.shape[1]-7) ] 
    resized_digit = cv2.cv2.resize(new_img, (18,18))
    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    

        
        
    return padded_digit.reshape(-1,IMG_SIZE,IMG_SIZE,1) 




model = tf.keras.models.load_model("neuralnet12.model")





blank = np.mean(prepare(nos[2]))



temp = []



for no in nos:
    if np.mean(prepare(no))>blank+1:
        prediction = model.predict([prepare(no)])
        # print(prediction)
        temp.append(np.argmax(prediction))
    else:
        temp.append(0)



print("\nRecognized Puzzle:\n")



# def print_grid(arr): 
#     for i in arr:
#         print(i)
  
template = np.reshape(temp,(-1,9))
template1=template.tolist()
# print(np.shape(template))

for i in template1:
    print(i)



def find_blank(arr, l): 
    for row in range(9): 
        for col in range(9): 
            if(arr[row][col]== 0): 
                l[0]= row 
                l[1]= col 
                return True
    return False
  

def row_check(arr, row, num): 
    for i in range(9): 
        if(arr[row][i] == num): 
            return True
    return False

def col_check(arr, col, num): 
    for i in range(9): 
        if(arr[i][col] == num): 
            return True
    return False
  
 
def box_check(arr, row, col, num): 
    for i in range(3): 
        for j in range(3): 
            if(arr[i + row][j + col] == num): 
                return True
    return False
  

def is_safe(arr, row, col, num): 
    
    return not row_check(arr, row, num) and not col_check(arr, col, num) and not box_check(arr, row - row % 3, col - col % 3, num) 
  

def solve_sudoku(arr): 
      
    
    l =[0, 0] 
         
    if(not find_blank(arr, l)): 
        return True
     
    row = l[0] 
    col = l[1] 
      
  
    for num in range(1, 10): 

        if(is_safe(arr, row, col, num)): 
              
            arr[row][col]= num 
  
            if(solve_sudoku(arr)): 
                return True
  
            arr[row][col] = 0
    
          
    return False 












# puzzle =[[3, 0, 6, 5, 0, 8, 4, 0, 0], 
#           [5, 2, 0, 0, 0, 0, 0, 0, 0], 
#           [0, 8, 7, 0, 0, 0, 0, 3, 1], 
#           [0, 0, 3, 0, 1, 0, 0, 8, 0], 
#           [9, 0, 0, 8, 6, 3, 0, 0, 5], 
#           [0, 5, 0, 0, 9, 0, 6, 0, 0], 
#           [1, 3, 0, 0, 0, 0, 2, 5, 0], 
#           [0, 0, 0, 0, 0, 0, 0, 7, 4], 
#           [0, 0, 5, 2, 0, 6, 3, 0, 0]] 


print("\nSolved Puzzle\n")

if(solve_sudoku(template1)): 
    for i in template1:
        print(i) 
else: 
    print("No solution exists")
