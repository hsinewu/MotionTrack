import numpy as np
import cv2

threshValue, blurValue = 20, 10
blurMode = True
# videoName = R"videos\oneStop.wmv"
videoName = R"videos\twoBack.wmv"
# videoName = R"videos\two.wmv"
# videoName = 0
cap = cv2.VideoCapture(videoName);
centers = [(0,0)]

# trackbar setup for parameter settings
def initTrackbar():
	cv2.namedWindow('panel')
	cv2.createTrackbar('thresh','panel',0,255,f)
	cv2.setTrackbarPos('thresh','panel',threshValue)

def f(v):
	global threshValue
	print (v)
	threshValue = v

# initTrackbar()

# init some frame
if cap.isOpened():
	_, cFrame = cap.read()
	cshape = cFrame.shape
	gPrev = cv2.cvtColor(cFrame, cv2.COLOR_BGR2GRAY)
	gshape = gPrev.shape
	colors = np.zeros(cshape)
	gBackground = gPrev
	# gBlackFrame = np.zeros(gshape)
	points = np.zeros(cshape)

while cap.isOpened():
	ret, cFrame = cap.read()
	if not ret:
		cap.open(videoName)
		continue
	cv2.imshow('capture', cFrame)

	# compute greyscale, calc difference, do threshold
	gFrame = cv2.cvtColor(cFrame, cv2.COLOR_BGR2GRAY)
	diff = cv2.absdiff(gPrev, gFrame)
	diffBackground = cv2.absdiff(gBackground, gFrame)
	# do I really want to blur thresh image?
	if blurMode:
		diff = cv2.blur(diff, (blurValue,blurValue))
		diffBackground = cv2.blur(diffBackground, (blurValue,blurValue))
	_, bDiff = cv2.threshold(diff, threshValue, 255, cv2.THRESH_BINARY)
	_, bDiffBg = cv2.threshold(diffBackground, threshValue, 255, cv2.THRESH_BINARY)

	# cv2.imshow('bDiff', bDiff)
	# cv2.imshow('bDiffBg', bDiffBg)

	# which diff mode to use
	useDiff = bDiff
	# useDiff = bDiffBg
	cv2.imshow('useDiff', useDiff)

	# draw the contours
	image, contours, hierarchy = cv2.findContours(useDiff.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	image = cv2.drawContours(image, contours, -1, 255, 1)
	cv2.imshow('coutour', image)

	cx, cy, M = 0, 0, 0
	# calc contours center
	if contours:	# test out of index
		# roi_points = contours[-1]	# heard that last one is of greatest area
		roi_points = reduce(lambda x,y: np.concatenate((x,y), axis=0), contours)	# No, lets assume that everything are interested
		M = cv2.moments(roi_points)
		roi_hull = cv2.convexHull(roi_points)
		bHull = np.zeros(gshape, np.uint8)
		cv2.fillPoly(bHull, [roi_hull], 255)
		cv2.imshow('bHull', bHull)
		roi_mask = roi_points
	# TODO: review how this center finding method works
	if M and M['m00']:	# test divide by zero
		cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
		centers.append((cx,cy))
	else:	# for some reason center point is not found
		continue

	# useMask = bDiff
	useMask = bHull
	
	points = cv2.addWeighted(points, 0.9, np.zeros(points.shape), 0.1, 0)
	cv2.circle(points, (cx,cy), 1, (255,255,255), -1)
	# cv2.imshow('center', points)
	predict = points.copy()
	predictPoint = (2*cx-centers[-2][0], 2*cy-centers[-2][1])
	cv2.circle(predict, predictPoint, 2, (0,255,0), -1)
	cv2.imshow('predict', predict)

	colors = cv2.addWeighted(colors, 0.9, np.zeros(colors.shape), 0.1, 0, dtype=cv2.CV_8U)	# TODO

	colorsOld = cv2.bitwise_and(colors, colors, mask=cv2.bitwise_not(useMask))	# TODO:why lose color:add?or?
	# cv2.imshow('colorsOld',colorsOld)
	colorsNew = cv2.bitwise_and(cFrame, cFrame, mask=useMask)
	# cv2.imshow('colorsNew',colorsNew)
	colors = cv2.add(colorsOld, colorsNew)

	cv2.imshow('colors',colors)

	gPrev = gFrame

	key = cv2.waitKey(10)
	if key == ord('q'):		# quit program
		break
	elif key == ord('z'):	# zerofy recorded points
		points.fill(0)
	elif key == ord('r'):	# read background frame
		gBackground = gFrame
	elif key == ord('b'):
		blurMode = not blurMode
	elif key == ord('p'):	# pause program
		while cv2.waitKey() != ord('p'):
			pass

cv2.destroyAllWindows();
cap.release()
