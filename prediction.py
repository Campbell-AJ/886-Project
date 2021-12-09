import matplotlib.pyplot as plt 
from fer import FER
import os


emo_detector = FER(mtcnn=True)


emotions_sum =  {'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}


directory = './'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        test_image_one = plt.imread(filename)
        captured_emotions = emo_detector.detect_emotions(test_image_one)
        dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
        if (dominant_emotion is not None):
            emotions_sum[dominant_emotion] = emotion_score + emotions_sum[dominant_emotion]
    else:
        continue


print(emotions_sum)

max_score = 0
max_emotion = None
for key, value in emotions_sum.items():
    if value > max_score:
        max_emotion = key
        max_score = value

print("Strongest emotion: " + max_emotion)
