import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture('/Users/fer/Downloads/VID-20190922-WA0105.mp4')

try:
    if not os.path.exists('/Users/fer/Desktop/VideoEnsayo'):
        os.makedirs('/Users/fer/Desktop/VideoEnsayo')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while True:
    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = '/Users/fer/Desktop/VideoEnsayo/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()