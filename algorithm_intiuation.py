import cv2
from deepface import DeepFace
from matplotlib import pyplot as plt
# Load the image
image_path = '2.jpg'
image = cv2.imread(image_path)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to RGB format
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # Extract the face ROI (Region of Interest)
    face_roi = rgb_image[y:y + h, x:x + w]

    # Perform emotion analysis on the face ROI
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    result_age = DeepFace.analyze(face_roi, actions=['age'], enforce_detection=False)

    # Determine the dominant emotion
    emotion = result[0]['dominant_emotion']
    age = result_age[0]['age']

    # Draw rectangle around face and label with predicted emotion
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, 'Emotion: ' + emotion, (x + 10, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(image, 'Age: ' + str(age), (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

# resized_image=cv2.resize(image,(700, 300))
# Display the image with detected faces and emotions
img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title('Resultant')
plt.show()
