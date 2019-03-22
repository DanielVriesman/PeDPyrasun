import numpy as np
import cv2

cap = cv2.VideoCapture(1)
#cap.set(3,1920)
#cap.set(4,1080)
cap.set(4,1280)
cap.set(3,720)
counter=1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    dst_img = np.zeros((frame.size, 3), np.uint8)

    center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
    # polar = cv2.logPolar(img,center,206,cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS,dst=dst_img)
    polar = cv2.logPolar(frame, center, 175, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS, dst=dst_img)

    cv2.imshow('polar',frame) #polar[:, 930:1080])

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
#    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        cv2.imwrite('Misumi-danificado2-polar/'+str(counter)+'.png',polar[:, 930:1080])
#        cv2.imwrite('frame'+str(counter) + '.png', frame)
        counter=counter+1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()