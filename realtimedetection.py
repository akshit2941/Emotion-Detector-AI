import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    # Assuming 'image' is a grayscale image
    feature = np.array(image)
    # Reshape the image to match the input shape expected by the model (1, height, width, channels)
    feature = feature.reshape(1, image.shape[0], image.shape[1], 1)
    # Normalize the pixel values to be between 0 and 1
    feature = feature / 255.0
    return feature

detected_emotions = []

# Change the argument to the path of your video file
# video_path = 'Emotion Detector/testvideo.mp4'
video_path = 'video.mp4'
#Your Path Of Video
video_capture = cv2.VideoCapture(video_path)

# video_capture = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            detected_emotions.append(prediction_label)
            cv2.putText(frame, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

video_capture.release()
cv2.destroyAllWindows()

# print("Detected Emotions:", detected_emotions)

emotion_counts = {label: detected_emotions.count(label) for label in labels.values()}
total_emotions = sum(emotion_counts.values())
emotion_percentages = {label: count / total_emotions * 100 for label, count in emotion_counts.items()}

# Print the percentage of each emotion
print("Emotion Percentages:")
for label, percentage in emotion_percentages.items():
    print(f"{label}: {percentage:.2f}%")
