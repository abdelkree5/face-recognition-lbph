import cv2
import numpy as np

# تحميل الموديل والأسماء
classifier = cv2.face.LBPHFaceRecognizer_create()
classifier.read('face_model.yml')
names = np.load('names.npy', allow_pickle=True).item()

# تحميل مصنف الوجه
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        cropped_face = image[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (200,200))
        return cropped_face, image

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    found_face, image = detect_face(frame)

    if found_face is not None:
        gray_face = cv2.cvtColor(found_face, cv2.COLOR_BGR2GRAY)
        label, score = classifier.predict(gray_face)

        confidence = int(100 * (1 - score / 400))
        name = names.get(label, "Unknown")

        text = f"{name} ({confidence}%)"
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
