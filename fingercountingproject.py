import cv2
import time
import HandTrackingModule as htm  # Importing custom hand tracking module
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for even fewer logs


# Open webcam
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480  # Set frame width and height
cap.set(3, wCam)       # Property 3 is width
cap.set(4, hCam)       # Property 4 is height

# Load all images from the folder 'fingerimages'
folderPath = "fingerimages"
myList = os.listdir(folderPath)
print(myList)  # Print list of image filenames

# Store all loaded images in overlayList
overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderPath}/{impath}')  # Read image
    overlayList.append(image)  # Add to list

print(len(overlayList))  # Print number of images loaded
pTime = 0  # Previous time for FPS calculation

# Initialize hand detector with detection confidence 0.75
detector = htm.hand(detectionCon=0.75)

# Landmark IDs of fingertips (thumb, index, middle, ring, pinky)
tipIds = [4, 8, 12, 16, 20]

# Start capturing frames in a loop
while True:
    success, img = cap.read()  # Read frame from webcam
    img = detector.findHands(img)  # Detect hands and draw landmarks
    lmList = detector.findPosition(img, draw=False)  # Get landmark positions

    if len(lmList) != 0:
        fingers = []

        # Check thumb (compare x-coordinates)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)  # Thumb is open
        else:
            fingers.append(0)  # Thumb is closed

        # Check other 4 fingers (compare y-coordinates)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)  # Finger is open
            else:
                fingers.append(0)  # Finger is closed

        totalFingers = fingers.count(1)  # Count how many fingers are open
        print(totalFingers)

        # Overlay corresponding image based on number of fingers
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        # Draw a rectangle to show number
        cv2.rectangle(img, (20, 255), (170, 425), (150, 150, 150), cv2.FILLED)
        # Put text (number of fingers) inside rectangle
        cv2.putText(img, str(totalFingers), (45, 400), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Show FPS on the screen
    # cv2.putText(img, f"FPS:{int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show final output frame
    cv2.imshow("Image", img)
    cv2.waitKey(1)
