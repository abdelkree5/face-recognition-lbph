import cv2
import numpy as np
import os

# تحميل مصنف الوجه
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# ادخل اسم الشخص اللي بتجمع له الصور
person_name = input("👤 اكتب اسم الشخص: ").strip()

# إنشاء فولدر التدريب لو مش موجود
data_path = 'trainingData/'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# فتح الكاميرا
capture = cv2.VideoCapture(0)
num_faces = 0
new_dimension = (200, 200)

print(f"📷 جاري التقاط صور {person_name}... اضغط Enter لإيقاف العملية أو انتظر 50 صورة.\n")

while True:
    ret, frame = capture.read()
    if not ret:
        print("❌ لم يتم فتح الكاميرا!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        num_faces += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, new_dimension)
        
        # خزن الصورة بالاسم
        file_name = f"{person_name}.{num_faces}.jpg"
        cv2.imwrite(os.path.join(data_path, file_name), face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Image {num_faces}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(100) == 13 or num_faces == 50:
        break

capture.release()
cv2.destroyAllWindows()
print(f"✅ تم حفظ صور {person_name} بنجاح!")
