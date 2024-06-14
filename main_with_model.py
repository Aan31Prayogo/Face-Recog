import cv2
import face_recognition
import pickle

#color
GREEN = (0,255,0)
RED = (0,0,255)
BLACK = (0,0,0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def main():
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (640,480))
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                color = GREEN
                name = known_names[best_match_index]
            else:
                color = RED

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), FONT, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__== '__main__':
    main()