from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
import cv2
import os
import codecs
from scipy import spatial
from scipy.special import softmax
import json
from PIL import Image
import numpy as np
from io import BytesIO
from json_models.response import Response
import mediapipe as mp

from models.resnet.model import ArtDLClassifier
from models.resnet.dataset import transform


app = FastAPI()

@app.post("/predict/", response_model=Response)
async def predict(file: UploadFile = File(...)):
    image_bytes=await file.read()
    stream = BytesIO(image_bytes)
    image = np.asarray(bytearray(stream.read()),dtype="uint8")
    image=cv2.imdecode(image,cv2.IMREAD_COLOR)
    cv2.imwrite('data/images/image.jpg', image)
    image = cv2.resize(image, (224, 224))
    if (isinstance(image,np.ndarray)):
        image =Image.fromarray(image)
    image = transform(image)

    # Stage 1: ResNet Classifier
    device = "cpu"
    model = ArtDLClassifier(num_classes = 2).to(device)
    model.load_state_dict(torch.load("models/resnet/artDLresnet50_224x224_2c_moredata_3.pt", map_location = device))
    model.eval()
    outputs = model(image.unsqueeze(0)).squeeze()
    print(outputs)
    out1 = np.argmax(outputs.detach().numpy())
    print(out1)
    ### If stage 1 predicts Mother mary we can skip stage 2 and 3 and go directly to stage 4

    # Stage 2: YOLO
    if out1 == 1:
        os.system("./bash.sh")
    # Stage 3: Word embeddings
    if out1 == 1:
        obj_text = codecs.open('models/embeddings/embeddings.json', 'r', encoding='utf-8').read()
        coordinates = json.loads(obj_text)

        mary = coordinates['mary']
        # Dictionary for distances from Mary
        l = {}
        for i in coordinates:
            l[i] = spatial.distance.cosine(mary, coordinates[i])

        with open('out2/exp/labels/'+imagename+'.txt', 'r') as f:
            lines = f.readlines()

        classes = ['baby','person','angel','book','jar','crown','bird','crescent','flowers','crucifict','pear','skull','lamb']
        
        present_in_image = []
        for line in lines:
            num = int(line.split(' ')[0])
            present_in_image.append(classes[num])

    
        avg_dist = 0
        for p in present_in_image:
            if p in l:
                avg_dist += l[p]
        avg_dist = avg_dist/len(present_in_image)

        if avg_dist < 1:
            out1=0     # Objects related to mother mary in the image, so we can switch the label to Mother Mary and continue the pipeline


    if out1 == 1:
        # Not mother mary
        return {
            'birth_virgin': 0 , 
            'marriage': 0, 
            'annunciation': 0, 
            'birth_jesus': 0, 
            'adoration': 0, 
            'coronation': 0, 
            'assumption': 0, 
            'death': 0, 
            'virgin_and_child': 0
        }
    
    # Stage 4: Pose estimation
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    mary_classes = ['birth_virgin', 'marriage', 'annunciation', 'birth_jesus', 'adoration', 'coronation', 'assumption', 'death', 'virgin_and_child']

    scores = np.zeros([9])
    imagename = 'image'
    if not os.path.exists('out2/exp/labels/'+imagename+'.txt'):
        print('out2/exp/labels/'+image+'.txt not present')
        scores = softmax(scores)
        scores = scores.tolist()
        return {
            'birth_virgin': scores[0] , 
            'marriage': scores[1], 
            'annunciation': scores[2], 
            'birth_jesus': scores[3], 
            'adoration': scores[4], 
            'coronation': scores[5], 
            'assumption': scores[6], 
            'death': scores[7], 
            'virgin_and_child': scores[8]
        }

    with open('out2/exp/labels/'+imagename+'.txt', 'r') as f:
        lines = f.readlines()

    img = cv2.imread('data/images/'+imagename+'.jpg')
    dh, dw, dc = img.shape
    present_in_image = []
    classes = ['baby','person','angel','book','jar','crown','bird','crescent','flowers','crucifict','pear','skull','lamb']
    coordinates = []
    for line in lines:
        # print(line)
        num = int(line.split(' ')[0])
        present_in_image.append(classes[num])
        class_id, x_center, y_center, w, h, prob = line.strip().split()
        x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
        x_center = round(x_center * dw)
        y_center = round(y_center * dh)
        w = round(w * dw)
        h = round(h * dh)
        x = round(x_center - w / 2)
        y = round(y_center - h / 2)
        coordinates.append([x, y, w, h])

    ### Scoring system
    if('person' in present_in_image and 'baby' in present_in_image):
        # add one to all the classes that have person and baby
        scores[0]+=1
        scores[3]+=1
        scores[4]+=1
        scores[8]+=1

    if(len(list(set(present_in_image))) < 4):
        scores[8]+=1
        
    if('crown' in present_in_image):
        scores[5]+=1
        scores[8]+=1

    if('bird' in present_in_image):
        scores[2]+=1
        scores[5]+=1
        scores[8]+=1

    if('lamb' in present_in_image):
        scores[3]+=1
        
    if('flowers' in present_in_image):
        scores[2]+=1
        scores[8]+=1

    if(present_in_image.count('person') > 2):
        scores[0]+=2
        scores[1]+=2
        scores[3]+=2
        scores[4]+=2
        scores[6]+=2
        scores[7]+=2

    if('jar' in present_in_image):
        scores[4]+=1

    if(len(present_in_image) < 4):
        scores[8]+=len(present_in_image)

    if(present_in_image.count('angel') > 0):
        scores[2]+=1
        scores[5]+=1
        scores[6]+=1

    if('book' in present_in_image):
        scores[2]+=1
        

    for i in range(len(coordinates)):
        if(present_in_image[i] == 'person'):
            x,y,w,h = coordinates[i]
            imgCrop = img[y:y + h, x:x + w]
            imgRGB = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_world_landmarks:
                # Nose
                nose = [results.pose_world_landmarks.landmark[0].x, results.pose_world_landmarks.landmark[0].y, results.pose_world_landmarks.landmark[0].z]


                # Right hand
                right_hand = [[results.pose_world_landmarks.landmark[16].x, results.pose_world_landmarks.landmark[16].y, results.pose_world_landmarks.landmark[16].z],
                            [results.pose_world_landmarks.landmark[18].x, results.pose_world_landmarks.landmark[18].y, results.pose_world_landmarks.landmark[18].z],
                            [results.pose_world_landmarks.landmark[20].x, results.pose_world_landmarks.landmark[20].y, results.pose_world_landmarks.landmark[20].z],
                            [results.pose_world_landmarks.landmark[22].x, results.pose_world_landmarks.landmark[22].y, results.pose_world_landmarks.landmark[22].z]
                ]


                # Left hand
                left_hand = [[results.pose_world_landmarks.landmark[15].x, results.pose_world_landmarks.landmark[15].y, results.pose_world_landmarks.landmark[15].z],
                            [results.pose_world_landmarks.landmark[17].x, results.pose_world_landmarks.landmark[17].y, results.pose_world_landmarks.landmark[17].z],
                            [results.pose_world_landmarks.landmark[19].x, results.pose_world_landmarks.landmark[19].y, results.pose_world_landmarks.landmark[19].z],
                            [results.pose_world_landmarks.landmark[21].x, results.pose_world_landmarks.landmark[21].y, results.pose_world_landmarks.landmark[21].z]
                ]

                right_shoulder = [results.pose_world_landmarks.landmark[12].x, results.pose_world_landmarks.landmark[12].y, results.pose_world_landmarks.landmark[12].z]

                left_shoulder = [results.pose_world_landmarks.landmark[11].x, results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z]

                avg_rh = np.mean(right_hand, axis = 0)

                avg_lh = np.mean(left_hand, axis = 0)
                r_horizontal_dist = abs(avg_rh[0] - right_shoulder[0])
                r_vertical_dist = abs(avg_rh[1] - right_shoulder[1])
                l_horizontal_dist = abs(avg_lh[0] - left_shoulder[0])
                l_vertical_dist = abs(avg_lh[1] - left_shoulder[1])

                ### if left hand and right hand are close together, most likely prayer
                dist = spatial.distance.cosine(avg_rh, avg_lh)
                if(dist < 0.01):
                    # print('prayer', dist)
                    scores[2]+=1
                    scores[4]+=1
                    scores[6]+=1
                    scores[7]+=1
                # Extended hand can be identified if hand is at a vertical and horizontal distance from the respective shoulder 
                elif((r_horizontal_dist > 0.15 and r_vertical_dist < 0.2) and (l_horizontal_dist > 0.15 and l_vertical_dist < 0.2)):
                    # print('both hands extended')
                    # similar to a T-pose
                    scores[6]+=1
                elif((r_horizontal_dist > 0.15 and r_vertical_dist > 0.2) or (l_horizontal_dist > 0.15 and l_vertical_dist > 0.2)):
                    # print('extended hand')
                    scores[1]+=1
                    scores[5]+=1
                # lying down - nose and shoulder on a similar height
                elif(abs(nose[1] - ((right_shoulder[1] +left_shoulder[1]) / 2)) < 0.05):
                    # print('lying_down')
                    scores[7]+=1
                    scores[0]+=1 #
                

    print(scores)
    scores = softmax(scores)
    ### need to delete image in data
    return {
        'birth_virgin': scores[0] , 
        'marriage': scores[1], 
        'annunciation': scores[2], 
        'birth_jesus': scores[3], 
        'adoration': scores[4], 
        'coronation': scores[5], 
        'assumption': scores[6], 
        'death': scores[7], 
        'virgin_and_child': scores[8]
    }
