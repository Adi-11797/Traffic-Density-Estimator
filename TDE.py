#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img1 = cv2.imread('Image1A.png', cv2.IMREAD_COLOR)
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('Image1B.png', cv2.IMREAD_COLOR)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create(1000) 

keypoints1, descriptors1 = orb.detectAndCompute(img1,None)
keypoints2, descriptors2 = orb.detectAndCompute(img2,None)



#bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
# matches = bf.knnMatch(descriptors1, descriptors2,k=2)



matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
good_points = []
good=[]
for m1, m2 in raw_matches:
    if m1.distance < 0.6 * m2.distance:
        good_points.append((m1.trainIdx, m1.queryIdx))
        good.append([m1])
img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, None, flags=2)


scale_percent = 100 # percent of original size
width = int(img3.shape[1] * scale_percent / 100)
height = int(img3.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img3 = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('Feature_Matched_Image.png', img3)


good1 = []
for m, n in raw_matches:
    if m.distance < 0.6 * n.distance:
        good1.append(m)


def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)


    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img


MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good1]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good1]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    
    result = warpImages(img2, img1, M)

    scale_percent = 100 # percent of original size
    width = int(img3.shape[1] * scale_percent / 100)
    height = int(img3.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    result = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
    
#     cv2.imshow("Result",result)
    cv2.imwrite('Stitched_Image.png', result)

cv2.waitKey()
cv2.destroyAllWindows()


# In[3]:


### Setting up YOLO for Object Detection ###

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

classes = None

with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


# In[5]:


### Vehicle Detection Using YOLO ###

img = cv2.imread("Stitched_Image.png")
dim = (800, 700)
  
# resize image
image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Defining Counters
bicycles = 0
cars = 0
motorcycles = 0
buses = 0
trucks = 0
areas = boxes
total_area = 0
road_area = Width*Height*0.85

for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    areas[i] = w*h*0.9
    if (class_ids[i] == 1):
        bicycles += 1
        total_area += areas[i]
    if (class_ids[i] == 2):
        cars += 1
        total_area += areas[i]
    if (class_ids[i] == 3):
        motorcycles += 1
        total_area += areas[i]
    if (class_ids[i] == 5):
        buses += 1
        total_area += areas[i]
    if (class_ids[i] == 7):
        trucks += 1
        total_area += areas[i]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

occupancy = (total_area/road_area)*100

print("pecentage occupancy: "+ "{}".format(occupancy))

cv2.imshow("object detection", image)
cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()

### Motion Detection using Image Subtraction ###

image1 = cv2.imread("Image1A.png")
image2 = cv2.imread("Image1B.png")

img_1 = cv2.imread("Image1A.png", 0)
img_2 = cv2.imread("Image1B.png", 0)

#REFERENCE IMAGE FOR SUBTRACTION
reference = img_1

#GRAYSCALE CONVERSION AND NOISE REMOVAL
background = cv2.GaussianBlur(reference, (21,21), 0)
gray = cv2.GaussianBlur(img_2, (21,21), 0)

#SUBTRACTION
subtraction = cv2.absdiff(background, gray)

#APPLICATION OF THRESHOLD
threshold = cv2.threshold(subtraction, 55, 255, cv2.THRESH_BINARY)[1]

#CONTOUR DETECTION 
contouring = threshold.copy()
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#REMOVING SMALL CONTOURS
for c in contours:
    if cv2.contourArea(c) < 200:  #removing smallest contours 
        continue
    (x,y,w,h) = cv2.boundingRect(c) #obtaining bounds of the contour 
  #drawing rectangle of bounds
#   cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('Feature Matched Image', img3)
cv2.imshow('Stitched Image', result)
cv2.imshow('image1', image1)
cv2.imwrite('motion1.png', image1)
cv2.imwrite('motion2.png', image2)
cv2.imshow('image2', image2)
# cv2.imshow('s',subtraction)
cv2.imshow('threshold', threshold)
cv2.imwrite('motion_flow.png', threshold)


cv2.waitKey()
cv2.destroyAllWindows()




