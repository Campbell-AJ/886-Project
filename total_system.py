import cv2 
import os
import matplotlib.pyplot as plt 
from fer import FER

filename = "video.mp4"

#capture video from webcam


# Open the device at the ID 0

cap = cv2.VideoCapture(0)

#Check whether user selected camera is opened successfully.

if not (cap.isOpened()):
    print("Could not open video device")

#To set the resolution

out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))

while True:
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) % 256 == 32:
        break


cap.release()
out.release()
cv2.destroyAllWindows()

#split video up into single frames

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




#use FER predictor to measure the emotion throughout the video based on summing prediction from single frames

emo_detector = FER(mtcnn=True)

emotions_sum =  {'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}


directory = './content'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        test_image_one = plt.imread('./content/' + filename)
        captured_emotions = emo_detector.detect_emotions(test_image_one)
        dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
        if (dominant_emotion is not None):
            emotions_sum[dominant_emotion] = emotion_score + emotions_sum[dominant_emotion]
    else:
        continue


max_score = 0
max_emotion = None
for key, value in emotions_sum.items():
    if value > max_score:
        max_emotion = key
        max_score = value

print("Strongest emotion: " + max_emotion)
