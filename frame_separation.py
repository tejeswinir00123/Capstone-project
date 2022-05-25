# Importing all necessary libraries
import cv2
import os
import datetime
import time
# Read the video from specified path

#os.chdir("C:/Users/keyesR/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/social_distancing")        
cam = cv2.VideoCapture("./Dataset 20 seconds/vlc-record-2022-04-10-21h20m53s-D02_20211216073908.mp4-.mp4")
#cam = cv2.VideoCapture(a)
frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(cam.get(cv2.CAP_PROP_FPS))
  
# calculate dusration of the video
seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))

##print("duration in seconds:", seconds)
##print("video time:", video_time)

try:
        
        # creating a folder named data
        if not os.path.exists('data1'):
                os.makedirs('data1')

# if not created then raise error
except OSError:
        print ('Error: Creating directory of data')

# frame
currentframe = 0
while(True):
        
        
        # reading from frame
        ret,frame = cam.read()
        #if ret and currentframe<=seconds:
        if ret and currentframe<=10:
                # Creating images in a loop
                name = './data1/Frame ' + str(currentframe) + '.jpg'
                print ('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe +=1
                time.sleep(1)
        else:
                break
os.system('py -3 first_frame.py --yolo yolo-coco')
cam.release()
cv2.destroyAllWindows()
