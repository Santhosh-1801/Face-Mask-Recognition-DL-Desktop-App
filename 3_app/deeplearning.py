import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import load_model  # type: ignore
import cv2  
from scipy.special import softmax

face_detection_model=cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt','./models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
model=tf.keras.models.load_model("face_cnn_model.h5")

labels=['Mask','No Mask','Covered Mouth Chin','Covered Nose Mouth']


def getColour(label):
    if label=='Mask':
        color=(0,255,0)
    elif label=='No Mask':
        color=(0,0,255)
    elif label=='Covered Mouth Chin':
        color=(0,255,255)
    else:
        color=(255,255,0)
    return color

def face_mask_prediction(img):
    image=img.copy()
    h,w=image.shape[:2]
    blob=cv2.dnn.blobFromImage(image,1,(300,300),(104,117,123),swapRB=True)
    
    face_detection_model.setInput(blob)
    detection=face_detection_model.forward()
    for i in range(0,detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.5:
            box=detection[0,0,i,3:7]*np.array([w,h,w,h])
            box=box.astype(int)
            pt1=(box[0],box[1])
            pt2=(box[2],box[3])
    
            face=image[box[1]:box[3],box[0]:box[2]]
            face_blob=cv2.dnn.blobFromImage(face,1,(100,100),(104,117,123),swapRB=True)
            face_blob_squeeze=np.squeeze(face_blob).T
            face_blob_rotate=cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            face_blob_flip=cv2.flip(face_blob_rotate,1)
    
    
            img_norm=np.maximum(face_blob_flip,0)/face_blob_flip.max()
            img_input=img_norm.reshape(1,100,100,3)
            result=model.predict(img_input)
            result=softmax(result)[0]
            confidence_index=result.argmax()
            confidence_score=result[confidence_index]
            label=labels[confidence_index]
            label_text='{}:{:,.0f} %'.format(label,confidence_score*100)
            color=getColour(label)
            cv2.rectangle(image,pt1,pt2,color,1)
            cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,2,color,1)

    return image
