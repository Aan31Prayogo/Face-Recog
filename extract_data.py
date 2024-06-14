import os
import face_recognition
import pickle
import sys


dataset_path = os.path.join(sys.path[0], "dataset")
encodings_path = "encodings.pickle"

# List to hold the encodings and labels
known_encodings = []
known_names = []

# Iterate over each person in the dataset
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue
    
    # Iterate over each image for the current person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        
        # Find the face encodings in the image
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_encodings.append(face_encodings[0])
            known_names.append(person_name)

# Save the encodings and names to a file
with open(encodings_path, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"Encodings saved to {encodings_path}")
