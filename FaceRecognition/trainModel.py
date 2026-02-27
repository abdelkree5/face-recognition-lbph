import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# مسار الصور
data_path = 'trainingData/'
files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []
names = {}
current_id = 0

for file in files:
    person_name = file.split('_')[0]  # الاسم قبل علامة "_"
    if person_name not in names.values():
        names[current_id] = person_name
        current_id += 1

    image_path = join(data_path, file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    Training_Data.append(np.asarray(img, dtype=np.uint8))
    label = list(names.keys())[list(names.values()).index(person_name)]
    Labels.append(label)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
model.write('face_model.yml')
np.save('names.npy', names)

print("✅ Model trained and saved successfully!")
print("🧾 Saved names:", names)
