import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imshow("Index 0", frame)
    cv2.waitKey(3000)
cap.release()
cv2.destroyAllWindows()
if cap.isOpened():
            print("[INFO] External camera detected, using it.")
            
else:
        print("[INFO] No external camera found, using built-in webcam.")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   


        
             