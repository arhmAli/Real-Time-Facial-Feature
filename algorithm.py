import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# List of image paths
image_paths = ['6.jpeg', '7.jpeg', '3.png', '4.jpeg', '5.jpeg']

# Create a figure for displaying results
fig, axs = plt.subplots(1, len(image_paths), figsize=(20, 10))
fig.suptitle('Resultant')

# Loop over each image
for idx, image_path in enumerate(image_paths):
    # Load the image
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

        # Draw rectangle around face and label with predicted emotion and age
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, 'Emotion: ' + emotion, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(image, 'Age: ' + str(age), (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Convert BGR image to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with detected faces and emotions in the subplot
    axs[idx].imshow(image_rgb)
    axs[idx].axis('off')
    axs[idx].set_title(f'Image {idx + 1}')

# Show the plot
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
