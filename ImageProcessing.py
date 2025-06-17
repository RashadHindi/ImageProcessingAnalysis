import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('breakfast.jpg')
cv.imshow("Display window", img)
k = cv.waitKey(0) 

#*******************************************************************#
src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Display window", src_gray)
k = cv.waitKey(0) 

#*******************************************************************#
dst = cv.equalizeHist(src_gray)
cv.imshow('Equalized Image', dst)
plt.hist(dst.ravel(),256,[0,256]); plt.show()
cv.waitKey(0)

#***********************************************************************#
c = np.random.uniform(0, 2)
print("\n")
print(f"The Value of c: {c}")
print(" \n")


modifiedImage = cv.convertScaleAbs(src_gray, alpha=c, beta=0)

saturation = (modifiedImage >= 255) & (modifiedImage <= 255)

highlightSaturation = cv.cvtColor(modifiedImage, cv.COLOR_GRAY2BGR)

highlightSaturation[saturation] = [39, 123, 183]
cv.imshow("Modified Grayscale Image with No Highlighting", modifiedImage)
cv.imshow("Modified Image with Saturation Highlighted", highlightSaturation)

cv.waitKey(0)

#******************************************************************************#
kernel = np.array([[1, 2, 2],
                       [1, 8, 0],
                       [2, 5, 5]], np.float32)

dst1 = cv.filter2D(src_gray, -1, kernel)
dst_normalized = cv.normalize(dst1, None, 0, 255, cv.NORM_MINMAX)
dst_normalized = dst_normalized.astype(np.uint8)
cv.imshow("Original gray scale Image", src_gray)
cv.imshow("Filtered Image", dst_normalized)
cv.waitKey(0)
#*****************************************************************************#
img_height, img_width = src_gray.shape

mask_h = 500
mask_w = 500

x = np.random.randint(0, img_width - mask_w)
y = np.random.randint(0, img_height - mask_h)

mask = np.zeros(src_gray.shape[:2], np.uint8)
mask[y:y+mask_h, x:x+mask_w] = 255  

masked_img = cv.bitwise_and(src_gray, src_gray, mask=mask)

plt.subplot(131), plt.imshow(src_gray, 'gray'), plt.title("Original Image")
plt.subplot(132), plt.imshow(mask, 'gray'), plt.title("Mask")
plt.subplot(133), plt.imshow(masked_img, 'gray'), plt.title("Masked Image")

plt.show()

#*********************************************************************************#
edges = cv.Canny(src_gray,100,200)
plt.subplot(121),plt.imshow(src_gray,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()
#**********************************************************************************#
slectPoint = []

def mouse_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        slectPoint.append((y, x))  
        cv.destroyAllWindows()


cv.imshow("Select a Seed Point", src_gray)
cv.setMouseCallback("Select a Seed Point", mouse_event)
cv.waitKey(0)

if not slectPoint:
    print("You didn't select a point")
    exit()

seed_y, seed_x = slectPoint[0]

try:
    intensity_range = int(input("Enter threshold: "))
except ValueError:
    print("Error: Invalid intensity range.")
    exit()

height, width = src_gray.shape
mask = np.zeros((height, width), dtype=np.uint8) 
seed_intensity = src_gray[seed_y, seed_x]

lower_bound = max(0, seed_intensity - intensity_range)
upper_bound = min(255, seed_intensity + intensity_range)

stack = [(seed_y, seed_x)]
mask[seed_y, seed_x] = 255

while stack:
    current_y, current_x = stack.pop()  
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_y, new_x = current_y + dy, current_x + dx

        if 0 <= new_y < height and 0 <= new_x < width:
            if mask[new_y, new_x] == 0:
                pixel_intensity = src_gray[new_y, new_x]
                if lower_bound <= pixel_intensity <= upper_bound:
                    mask[new_y, new_x] = 255  
                    stack.append((new_y, new_x))  

cv.imshow("Region Growing Result", mask)
cv.waitKey(0)
cv.destroyAllWindows()