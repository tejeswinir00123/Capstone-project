# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import imutils
import csv
import glob
from csv import reader
import math
import pandas as pd
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import seaborn as sn
import time


count=0
count1=0
count2=0
count3=0


def detect(frame):
        global count
        person = 0
        #newfile='Frame' +str(count)
        newfile='./Bounding Box Coordinates/Frame0'
        myfile = open("%s.csv" %newfile, 'w',newline='')
       
        ap = argparse.ArgumentParser()
       
        ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
        ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applying non-maxima suppression")
        args = vars(ap.parse_args())
       
        labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")

       
        weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
        configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
       
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        image = imutils.resize(frame, width = min(1100, frame.shape[1]))
        (H, W) = image.shape[:2]
       
        ln = net.getLayerNames()
        ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (316, 316),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
                
                for detection in output:
                       
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                      
                        if confidence > args["confidence"]:
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])
        topleftlist=[]
        bottomrightlist=[]
        distance_camera=[]
        if len(idxs) > 0:
                
                for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                       
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        writer=csv.writer(myfile,delimiter=',')
                        writer.writerow(["Person %d" % person,"%d" % (x) ,"%d" % (y)])
                        cv2.putText(image, f'Person {person}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                        person += 1
                        #cv2.putText(image,(x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        cv2.imshow("Image", image)
        count+=1
        cv2.waitKey(0)

def distance():
    global count1
    #newfile1='Frame' +str(count1)
    newfile1='./Bounding Box Coordinates/Frame0'
    with open("%s.csv" %newfile1,newline='') as file:
        reader=csv.reader(file,delimiter=',')
        Output=[]
        Output1=[]
        for row in reader:
            Output.append(row[1:])
            Output1.append(row[1:])
    Output = [[int(float(j)) for j in i] for i in Output]
    Output1 = [[int(float(j)) for j in i] for i in Output1]
    final_list=[]
    for a in Output1:
        temp = []
        for b in Output:
            dis = sum([pow(a[i] - b[i], 2) for i in range(len(a))])
            temp.append(round(pow(dis, 0.5),4))
        final_list.append(temp)
    #newfile2='Frame '+str(count1)+' Euclidean Distance'
    newfile2='./Euclidean distance/Frame 0 Euclidean Distance'
    df = pd.read_csv("%s.csv" %newfile1,header=None)
    first_column = df.iloc[:, 0]
    with open("%s.csv" %newfile2,'w',newline='') as file1:
        writer1=csv.writer(file1)
        writer1.writerows(zip(first_column,Output,Output1,final_list))
    count1+=1


def calculate_contact():
    count4=0
    count6=0
    newfile2='./Euclidean distance/Frame '+str(count4)+' Euclidean Distance'
    df = pd.read_csv("%s.csv" %newfile2,header=None)
    df[3] = df[3].apply(eval)
    k=0
    a=[]
    b=[]
    b1=[]
    new_row=pd.Series()
    df4=pd.DataFrame()
    for i in range(len(df)):
        final_list=[]
        final_list1=[]
        c={}
        e=df[0][i]
        b.append(e)
        #print("Contact Details for {}".format(e))
        for j in df[3][i]:
            if df[3][i][k]>0 and df[3][i][k]<80:
                f=df[0][k]
                c[f]=abs(df[3][i][k])
                final_list.append(f)
                #print(f)
            k=k+1
        final_list1.append(c)
        b1.append(final_list1)
        a.append(final_list)
        if len(final_list)>0:
            newfile1='./Close Contacts/Frame '+str(count4)+' Contact Details'
            myfile = open("%s.csv" %newfile1, 'w',newline='')
            writer=csv.writer(myfile)
            writer.writerows(zip(b,b1,a))
            #k=0
            count6=count6+1
        k=0
    if count6>0:
        df2=pd.DataFrame(b)
        df1 = pd.DataFrame(a)
        new_row=pd.Series(a)
        df4=df4.append(new_row,ignore_index=True)
        df4=pd.melt(df4)
        unique_items = to_1D(df4['value']).value_counts()
        persons_bool = boolean_df(df4['value'], unique_items.keys())
        persons_int = persons_bool.astype(int)
        persons_freq_mat = np.dot(persons_int.T, persons_int)
        persons_freq = pd.DataFrame(persons_freq_mat, columns = unique_items.keys(), index = unique_items.keys())
        fig, ax = plt.subplots(figsize = (13,7))
        sn.heatmap(persons_freq, cmap = "rainbow")
        plt.xticks(rotation=50)
        plt.savefig("./HeatMaps/heatmap{}.png".format(count4), dpi = 300)
    else:
        print("No close contacts")




def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


def boolean_df(item_lists, unique_items):
    bool_dict = {}
    for i, item in enumerate(unique_items):
        bool_dict[item] = item_lists.apply(lambda x: item in x)
    return pd.DataFrame(bool_dict)


        
def detectByPathImage():
    try:
       if not os.path.exists('Bounding Box Coordinates'):
            os.makedirs('Bounding Box Coordinates')
       if not os.path.exists('Euclidean distance'):
            os.makedirs('Euclidean distance')
       if not os.path.exists('Close Contacts'):
            os.makedirs('Close Contacts')
       if not os.path.exists('HeatMaps'):
            os.makedirs('HeatMaps')
    except OSError:
        print ('Error: Creating directory of data')
    path='./Frames/Frame 0.jpg'
    image = cv2.imread(path)
    #image = imutils.resize(image, width = min(800, image.shape[1]))
    detect(image)
    distance()
    calculate_contact()
    os.system('python -i yolo.py --yolo yolo-coco')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectByPathImage()

	
