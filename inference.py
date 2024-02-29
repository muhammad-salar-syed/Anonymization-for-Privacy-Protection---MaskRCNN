from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import cv2
import cvzone
from matplotlib.patches import Rectangle


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "person_cfg"
	# number of classes
	NUM_CLASSES = 1 + 2
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

cap = cv2.VideoCapture('./person.mp4')
ret, frame = cap.read()
H, W, _ = frame.shape
# out = cv2.VideoWriter('./out_person.mp4', cv2.VideoWriter_fourcc(*'mpv4'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_persons_cfg_0048.h5', by_name=True)


while ret:
    
    #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    detected = model.detect([frame])[0]
    class_id_counter=1
    class_names = ['man', 'women']
    for box in detected['rois']:
        #print(box)
        #get coordinates
        detected_class_id = detected['class_ids'][class_id_counter-1]

        if class_names[detected_class_id-1]=='women':
            y1, x1, y2, x2 = box
            w, h = x2 - x1, y2 - y1
            roi = frame[y1:y1+h, x1:x1+w]
            blur_image = cv2.GaussianBlur(roi,(51,51),0)
            frame[y1:y1+h, x1:x1+w] = blur_image
            cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)),l=15,t=2,colorR=(0,0,0),colorC=(0,0,255))
            cv2.putText(frame, class_names[detected_class_id-1].upper(), (int(x1), int(y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            #print(detected_class_id)
            #print("Detected class is :", class_names[detected_class_id-1])
            y1, x1, y2, x2 = box
            #calculate width and height of the box
            w, h = x2 - x1, y2 - y1
            #create the shape
            cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)),l=15,t=2,colorR=(0,0,0))
            #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, class_names[detected_class_id-1].upper(), (int(x2 - 25), int(y2 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        class_id_counter+=1

    #cv2.imshow('image',cv2.resize(frame,(800,800)))
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    #out.write(frame)
    ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()