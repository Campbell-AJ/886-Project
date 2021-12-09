import cv2
import os
vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0


directory = './content'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        os.remove('./content/' + filename) 
    else:
        continue

while success:
  cv2.imwrite("./content/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
