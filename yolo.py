# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os,sys
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

count=1
count1=1
count2=1
count3=0
count4=1

def detect(frame):
    global count
    global count3
    person = 0
    col1=1
    col2=2
    row=0
    i=0
    count_first=1
    new_id=0
    newfile1='./Bounding Box Coordinates/Frame' +str(count3)
    df = pd.read_csv("%s.csv" %newfile1,header=None)
    #print(df)
    df1=df[0].str.slice(7,8)
    arraynew=df1.to_numpy()
    #print(arraynew)
    arraynew1=np.array(arraynew)
    df3=df[1]
    arraynew2=np.array(df3)
    #print(list(map(int, arraynew2)))
    df4=df[2]
    arraynew3=np.array(df4)
    df2=df.set_index(df1)

    ### construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ##ap.add_argument("-i", "--image", required=True,help="path to input image")
    ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensios
    #image = cv2.imread(args["image"])
   
    #image = cv2.imread(frame,0)
    image = imutils.resize(frame, width = min(1100, frame.shape[1]))
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (316, 316),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
     # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
    newfile='./Bounding Box Coordinates/Frame' +str(count)
    myfile = open("%s.csv" %newfile, 'w',newline='')
    topleftlist=[]
    bottomrightlist=[]
    # ensure at least one detection exists
    if len(idxs) > 0:
            # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            topleftlist=topleftlist+[(x)]
            bottomrightlist=bottomrightlist+[(y)]
    boundingboxlist=list(zip(topleftlist,bottomrightlist))
    boundingboxlist
    boundingboxdf=pd.DataFrame(boundingboxlist)
    boundingboxdf=np.ceil(boundingboxdf).astype('int')
    #print(boundingboxdf)
    bounding_col0=boundingboxdf[0]
    bounding_col1=boundingboxdf[1]
    bounding_arr0=np.array(bounding_col0)
    bounding_arr1=np.array(bounding_col1)
##    print(bounding_arr0)
##    print(bounding_arr1)
    topleftlist.sort()
    dftopleftlist=pd.DataFrame(topleftlist)
    dfbottomrightlist=pd.DataFrame(bottomrightlist)
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        if len(topleftlist)==len(arraynew2) or len(topleftlist)<len(arraynew2):
            a=df.iat[row,col1]
            b=df.iat[row,col2]
            index_val=[index for index in range(len(arraynew2)) if arraynew2[index] == a]
            c1=boundingboxdf.iat[row,0]
            #print(c1)
            index_val1=[arraynew2[index] for index in range(len(arraynew2)) if c1-arraynew2[index] <=3]
            min_index_val1=min(index_val1)
            #print(min_index_val1)
            d1=boundingboxdf.loc[boundingboxdf[0]==c1,1].iloc[0]
            #print(d1)
            e1=boundingboxdf.loc[boundingboxdf[1]==d1,0].iloc[0]
            #print(e1)
            index_val2=[arraynew3[index] for index in range(len(arraynew3)) if d1-arraynew3[index] <=3]
            #print(index_val2)
            min_index_val2=min(index_val2)
            #print(min_index_val2)
            c=e1-min_index_val1
            d=d1-min_index_val2
            
            if min_index_val2:
                if c<=3 and d<=3:
                    e=df.loc[df[1]==min_index_val1,0].iloc[0]
                    #print(e)
                    cv2.putText(image, f' {e}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer=csv.writer(myfile,delimiter=',')
                    writer.writerow([f'{e}',"%d" %  e1,"%d" %d1])
                    row=row+1
            else:
                if count_first==1:
                    df1=df[0].str.slice(7,8)
                    arraynew=df1.to_numpy()
                    new_person_id=len(topleftlist)+1
                    cv2.putText(image, f' {new_person_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer=csv.writer(myfile,delimiter=',')
                    #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                    writer.writerow([f'{new_person_id}',"%d" %  e1,"%d" %d1])
                    row=row+1
                    count_first=count_first+1
                    new_id=max_new_person_id
                else:
                    new_id=new_id+1
                    #print(new_id)
                    cv2.putText(image, f' {new_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer=csv.writer(myfile,delimiter=',')
                    #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                    writer.writerow([f'{new_id}',"%d" %  e1,"%d" %d1])
                    row=row+1
                        
        else:
            new_c1=boundingboxdf.iat[row,0]
            #print(new_c1)
            d1=boundingboxdf.loc[boundingboxdf[0]==new_c1,1].iloc[0]
            #print(d1)
            e1=boundingboxdf.loc[boundingboxdf[1]==d1,0].iloc[0]
            #print(e1)
            index_value=[arraynew2[index] for index in range(len(arraynew2)) if new_c1-arraynew2[index] <=3]
            #print(index_value)
            if row<len(arraynew2) and not index_value:        
                a=df.iat[row,col1]
                b=df.iat[row,col2]
                index_val=[index for index in range(len(arraynew2)) if arraynew2[index] == a]
                c1=boundingboxdf.iat[row,0]
                index_val1=[arraynew2[index] for index in range(len(arraynew2)) if c1-arraynew2[index] <=3]
                #print(index_val1)
                if not index_val1:
                    if count_first==1:
                        new_person_id=len(topleftlist)+1
                        cv2.putText(image, f' Person {new_person_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        writer=csv.writer(myfile,delimiter=',')
                        #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                        writer.writerow([f'Person {new_person_id}',"%d" %  e1,"%d" %d1])
                        row=row+1
                        count_first=count_first+1
                        new_id=new_person_id
                    
                    else:
                        new_id=new_id+1
                        #print(new_id)
                        cv2.putText(image, f' Person {new_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        writer=csv.writer(myfile,delimiter=',')
                        #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                        writer.writerow([f'Person {new_id}',"%d" %  e1,"%d" %d1])
                        row=row+1
                else:
                    min_index_val1=min(index_val1)
                    ##print(min_index_val1)
                    d1=boundingboxdf.loc[boundingboxdf[0]==c1,1].iloc[0]
                    e1=boundingboxdf.loc[boundingboxdf[1]==d1,0].iloc[0]
                    ##print(d1)
                    index_val2=[arraynew3[index] for index in range(len(arraynew3)) if d1-arraynew3[index] <=3]
                    min_index_val2=min(index_val2)
                    c=e1-min_index_val1
                    #d=d1-b
                    d=d1-min_index_val2
                    if c>=0 and c<=3 and d>=0 and d<=3:
                        e=df.loc[df[1]==min_index_val1,0].iloc[0]
                        #print(e)
                        cv2.putText(image, f' {e}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        writer=csv.writer(myfile,delimiter=',')
                        #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                        writer.writerow([f' {e}',"%d" %  e1,"%d" %d1])
                        row=row+1
                        #print(row)
            elif row<len(arraynew2) and index_value:
                a=df.iat[row,col1]
                b=df.iat[row,col2]
                index_val=[index for index in range(len(arraynew2)) if arraynew2[index] == a]
                c1=boundingboxdf.iat[row,0]
                index_val1=[arraynew2[index] for index in range(len(arraynew2)) if c1-arraynew2[index] <=3]
                min_index_val1=min(index_val1)
                d1=boundingboxdf.loc[boundingboxdf[0]==c1,1].iloc[0]
                e1=boundingboxdf.loc[boundingboxdf[1]==d1,0].iloc[0]
                index_val2=[arraynew3[index] for index in range(len(arraynew3)) if d1-arraynew3[index] <=3]
                min_index_val2=min(index_val2)
                c=e1-min_index_val1
                #print(e1)
                #print(min_index_val1)
                #d=d1-b
                d=d1-min_index_val2
                #print(d1)
                #print(min_index_val2)
                if c>=-5 and c<=3 and d<=3:
                    e=df.loc[df[1]==min_index_val1,0].iloc[0]
                    #print(e)
                    cv2.putText(image, f'{e}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer=csv.writer(myfile,delimiter=',')
                    #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                    writer.writerow([f'{e}',"%d" %  e1,"%d" %d1])
                    row=row+1
                else:
                    df1=df[0].str.slice(7,8)
                    arraynew=df1.to_numpy()
                    if count_first==1:
                        new_person_id=len(topleftlist)+1
                        cv2.putText(image, f' Person {new_person_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        writer=csv.writer(myfile,delimiter=',')
                        #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                        writer.writerow([f'Person {new_person_id}',"%d" %  e1,"%d" %d1])
                        row=row+1
                        count_first=count_first+1
                        new_id=new_person_id
                    else:
                        new_id=new_id+1
                        #print(new_id)
                        cv2.putText(image, f' Person {new_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        writer=csv.writer(myfile,delimiter=',')
                        #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                        writer.writerow([f'Person {new_id}',"%d" %  e1,"%d" %d1])
                        row=row+1
            else:
                df1=df[0].str.slice(7,8)
                arraynew=df1.to_numpy()
                if count_first==1:
                    new_person_id=len(topleftlist)+1
                    cv2.putText(image, f' Person {new_person_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer=csv.writer(myfile,delimiter=',')
                    #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                    writer.writerow([f'Person {new_person_id}',"%d" %  e1,"%d" %d1])
                    row=row+1
                    count_first=count_first+1
                    new_id=new_person_id
                else:
                    new_id=new_id+1
                    #print(new_id)
                    cv2.putText(image, f' Person {new_id}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer=csv.writer(myfile,delimiter=',')
                    #writer.writerow([f'{e}',"%d" %  boundingboxdf.iat[row,0],"%d" % boundingboxdf.iat[row,col1]])
                    writer.writerow([f'Person {new_id}',"%d" %  e1,"%d" %d1])
                    row=row+1
    #cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    #cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

    cv2.imshow('output', image)
    #name = './Frame.jpg'
    #cv2.imwrite(name , frame)
    count+=1
    count3+=1
    return image
    time.sleep(0)           
                                                                    
def distance():
    global count1
    newfile1='./Bounding Box Coordinates/Frame' +str(count1)
    with open("%s.csv" %newfile1,newline='') as file:
        reader=csv.reader(file,delimiter=',')
        Output=[]
        Output1=[]
        for row in reader:
            Output.append(row[1:])
            Output1.append(row[1:])
    Output = [[int(float(j)) for j in i] for i in Output]
    Output1 = [[int(float(j)) for j in i] for i in Output1]
##    print(Output)
##    print(Output1)
    final_list=[]
    for a in Output1:
        temp = []
        for b in Output:
            dis = sum([pow(a[i] - b[i], 2) for i in range(len(a))])
            temp.append(round(pow(dis, 0.5),4))
        final_list.append(temp)
##    print(final_list)
    newfile2='./Euclidean distance/Frame '+str(count1)+' Euclidean Distance'
    df = pd.read_csv("%s.csv" %newfile1,header=None)
    first_column = df.iloc[:, 0]
    with open("%s.csv" %newfile2,'w',newline='') as file1:
        writer1=csv.writer(file1)
        writer1.writerows(zip(first_column,Output,Output1,final_list))
    count1+=1


def calculate_contact():
    global count4
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
    count4+=1


def summary_report():
    unique_items = set()
    count2=0
    df3=pd.DataFrame()
    df4=pd.DataFrame()
    df5=pd.DataFrame()
    while(True) and count2<9:
        path='./Close Contacts/Frame ' + str(count2) + ' Contact Details'+'.csv'
        path1='./Close Contacts/Frame ' + str(count2+1) + ' Contact Details'+'.csv'
        df4=pd.read_csv(path,header=None)
        df4[2][0] = df4[2][0].replace(', ', '","')
        df4[2][0] = df4[2][0].replace('[', '["')
        df4[2][0] = df4[2][0].replace(']', '"]')
        df4[2]=df4[2].apply(eval)
        d= dict([(i,b) for i,b in zip(df4[0],df4[2])])
        df5=pd.read_csv(path1,header=None)
        df5[2][0] = df5[2][0].replace(', ', '","')
        df5[2][0] = df5[2][0].replace('[', '["')
        df5[2][0] = df5[2][0].replace(']', '"]')
        df5[2]=df5[2].apply(eval)
        d1= dict([(i,b) for i,b in zip(df5[0],df5[2])])
        #print(d1)
        count2=count2+1
        res = dict()
        for key in d: 
            if key in d1: 
                res[key] = []
                for val in d[key]:
                    if val in d1[key]:
                        res[key].append(val)
    newfile='./Summary Report/Summary Report.csv'
    with open(newfile,'w',newline='') as file1:
        writer=csv.writer(file1,delimiter=',')
        for key,value in res.items():
            writer.writerow([key,value])

def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


def boolean_df(item_lists, unique_items):
    bool_dict = {}
    for i, item in enumerate(unique_items):
        bool_dict[item] = item_lists.apply(lambda x: item in x)
    return pd.DataFrame(bool_dict)

    
def detectByPathImage():
    global count2
    
    try:
        if not os.path.exists('Bounding Box Coordinates'):
            os.makedirs('Bounding Box Coordinates')
        if not os.path.exists('Euclidean distance'):
            os.makedirs('Euclidean distance')
        if not os.path.exists('Close Contacts'):
            os.makedirs('Close Contacts')
        if not os.path.exists('HeatMaps'):
            os.makedirs('HeatMaps')
        if not os.path.exists('Summary Report'):
            os.makedirs('Summary Report')
    except OSError:
        print ('Error: Creating directory of data')

    while(True):
        path='./Frames/Frame ' + str(count2) + '.jpg'
        print(count2)
        image = cv2.imread(path)
        #image = imutils.resize(image, width = min(1100, image.shape[1]))
        count2+=1
        if count2%11==0 and count2==11:
           detect(image)
           distance()
           dir = os.listdir('./Close Contacts')
           if len(dir) != 0:
               calculate_contact()
               summary_report()
           else:
               print("No close contacts hence no summary report")
        else:
            detect(image)
            distance()
            calculate_contact()
        #os.system('python contact_details.py')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detectByPathImage()

	
