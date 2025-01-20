import cv2

image = cv2.imread('kucing.jpg') 
orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(image, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# menggunakan VcXsrv
cv2.imshow("Kucing Lucu", image_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
