import cv2
import face_recognition
import os, sys
import time
import threading
import time
from dotenv import load_dotenv

load_dotenv()

# Path to the known image
IMAGE_NAME = os.getenv('IMAGE_NAME')
PATH_IMAGE = os.path.join(sys.path[0] + "/" +IMAGE_NAME)
TIMER = 5

#load model
known_image = face_recognition.load_image_file(PATH_IMAGE)
known_encoding = face_recognition.face_encodings(known_image)[0]

#flaging
flag_process_frame = True
flag_stop = False

#color
GREEN = (0,255,0)
RED = (0,0,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def start_timer_flag():
    threading.Thread(target=timer_flag).start()
    
def timer_flag():
    global flag_process_frame
    global flag_stop
    global TIMER
    
    while True:
        TIMER-=1
        time.sleep(1)
        
        if flag_stop:
            flag_stop = False
            print('[DEBUG] STOP FLAG TIMER')
            break
        
        if TIMER==0:
            TIMER = 5
            flag_process_frame = True
            print("[INFO] reset capture flag", flag_process_frame)

def main():
    global flag_process_frame
    global flag_stop
    status = ""
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if flag_process_frame:
            print("[INFO] start process image")
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([known_encoding], face_encoding)
                if True in matches:
                    print("[DEBUG] Valid person")
                    status = "VALID PERSON"
                    color = GREEN
                else:
                    print("[DEBUG] Invalid person")
                    status = "INVALID PERSON"
                    color = RED
            flag_process_frame = False

        cv2.putText(frame, status, (10,40), FONT, 1, color, 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            flag_stop = True
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_timer_flag()
    main()