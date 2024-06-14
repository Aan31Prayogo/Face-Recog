import cv2
import face_recognition
import os, sys
from dotenv import load_dotenv

load_dotenv()
IMAGE_NAME = os.getenv('IMAGE_NAME')
PATH_IMAGE = os.path.join(sys.path[0] + IMAGE_NAME)
known_image = face_recognition.load_image_file(PATH_IMAGE)
known_encoding = face_recognition.face_encodings(known_image)[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        if True in matches:
            print("Valid person")
        else:
            print("Invalid person")

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
