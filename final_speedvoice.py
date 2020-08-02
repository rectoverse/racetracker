import numpy as np
import time
import cv2
import os
import imutils
from gtts import gTTS 
from playsound import playsound

def compare(img1,img2): 
	img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
	img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
 
	hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
	cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
	hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
	cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

	metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
	return metric_val

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('output_subtitle_commentry.avi', fourcc, 15.0, (640, 480))

vs = cv2.VideoCapture('speed_out.avi')
(W, H) = (None, None)
count = 0

ref=[]	
pos=[]
language = 'en'



for i in os.listdir('ref'):
	ref.append('ref/'+i)

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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

			if confidence > 0.5:
			
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
	
	#cv2.line(frame, (0,350), (1500,350), (0,0,0), 2)
	#cv2.line(frame, (0,370), (1500,370), (0,0,0), 2)
	
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if LABELS[classIDs[i]] == 'car':
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)					
				if y>= 350 and y <= 370:
					img=frame[y:y+h, x:x+w]

					e=1
					r=''
					for j in ref:
						#print(i,j)
						ref_img = cv2.imread(j)
						val = compare(img,ref_img)
						#print(val)
						if e>val:
							e=val
							r=j
					#print(r)
					rr =r.split('/')
					rr=rr[1].split('.')
					if rr[0] not in pos:
						pos.append(rr[0])
						#print(rr[0])
						txt='race car {} is at position {}'.format(rr[0],len(pos))
						if len(pos)==1:	
							file=open('winner.txt','w')
							file.write(str(rr[0]))
							file.close()
						#print(txt)
						
						#cv2.putText(frame, txt, (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 
						myobj = gTTS(text=txt, lang=language, slow=False)
						myobj.save("commentry/commentry{}.mp3".format(len(pos)))
						playsound("commentry/commentry{}.mp3".format(len(pos)))
					
			count += 1
	
	
	cv2.imshow('test',frame)
	frame = cv2.resize(frame, (640,480))
	out.write(frame)
	if cv2.waitKey(1) &0XFF == ord('x'):
		break

		
vs.release()
out.release() 
cv2.destroyAllWindows() 