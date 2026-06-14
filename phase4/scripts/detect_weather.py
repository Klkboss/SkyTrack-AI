import cv2
import numpy as np

VIDEO = r"C:\DroneProject\phase4\inputs\test_video.mp4"

cap = cv2.VideoCapture(VIDEO)

brightness_values = []
blue_values = []

while True:

    ret,frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])

    blue_channel = np.mean(frame[:,:,0])

    brightness_values.append(brightness)
    blue_values.append(blue_channel)

cap.release()

avg_brightness = np.mean(brightness_values)
avg_blue = np.mean(blue_values)

weather = "Unknown"

if avg_brightness > 170:
    weather = "Sunny"
elif avg_brightness > 120:
    weather = "Cloudy"
else:
    weather = "Dark / Rainy"

print("Detected Weather:",weather)