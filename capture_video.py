import cv2 

filename = "video.mp4"


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



