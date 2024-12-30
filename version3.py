from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize audio mixer
mixer.init()
mixer.music.load("C:\\Users\\Huawei\\Downloads\\drowsiness youtube\\music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to release resources
def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()

# Function to display alert text
def display_alert(frame):
    alert_text = "****************ALERT!****************"
    cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    alert_text1 = "********DROWSINESS DETECTED!*********"
    cv2.putText(frame, alert_text1, (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# Function to process each frame
def process_frame(frame, flag):
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = calculate_ear(leftEye)
        rightEAR = calculate_ear(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize eyes
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < THRESHOLD_EAR:
            flag += 1
            if flag >= FRAME_CHECK_LIMIT:
                print('DROWSY: Reducing Speed')
                display_alert(frame)
                if not mixer.music.get_busy():
                    mixer.music.play()
        else:
            flag = 0
            print('AWAKE')

    return frame, flag

# Constants
THRESHOLD_EAR = 0.25
FRAME_CHECK_LIMIT = 20

# Initialize face detection and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Huawei\\Downloads\\drowsiness youtube\\models/shape_predictor_68_face_landmarks.dat")

# Facial landmark indices for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Main function
def main():
    cap = cv2.VideoCapture(0)
    flag = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, flag = process_frame(frame, flag)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        cleanup(cap)
        print("Resources released. Program terminated.")

if __name__ == "__main__":
    main()
