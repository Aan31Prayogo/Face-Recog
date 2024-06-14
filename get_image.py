import cv2
import argparse
import os
import sys

PATH_DATASET = os.path.join(sys.path[0], "dataset")
cap = cv2.VideoCapture(0)
LIMIT = 100

def check_folder_dataset_exist():
    if not os.path.exists(PATH_DATASET):
        print("[DEBUG] folder dataset is NOT exist")
        os.mkdir(PATH_DATASET)
    else:
        print("[DEBUG] folder dataset is exist")

def get_name_args()-> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="must be a string")
    args = parser.parse_args()
    name = args.name 
    return name

def main(name: str):
    faces = None
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640,480))
        if not ret:
            print("[WARNING] Failed to open camera")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            img = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

        key =  cv2.waitKey(1) & 0xFF
            
        #quit video capture
        if key==ord('q'):
            print("[INFO] Close by button")
            break    

        # #capture image 
        if key==ord('c'):
            print("[INFO] Capture by button")
            
            #check path person
            path_person = PATH_DATASET + "/" + name
            if not os.path.exists(path_person):
                os.mkdir(path_person)
                
            if faces is not None:
                for i in range(LIMIT + 1):
                    filename_ = f"/image{i+1}.jpg"
                    cv2.imwrite(path_person + filename_, frame)
                break
        
        #cv2.imshow("frame", frame)
        cv2.imshow("frame", gray)

    
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    check_folder_dataset_exist()
    name = get_name_args()
    main(name=name)
        