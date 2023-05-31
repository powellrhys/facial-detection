import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('sample/wales_fa.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Resize the sample image for display
scale_factor = 90
height = int(img.shape[0] * scale_factor / 100)
width = int(img.shape[1] * scale_factor / 100)
img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()