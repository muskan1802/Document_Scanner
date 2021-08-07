import cv2
import numpy as np

hi=640
wi = 540
# hi=320
# wi = 480
kernel = np.ones((5,5))

cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(3,320)
# cap.set(4,480)
cap.set(10,150)

def preprocessing(img):
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray,(5,5),1)
    imgcanny = cv2.Canny(imgblur,200,200)
    imgDiala = cv2.dilate(imgcanny ,kernel,iterations=2)
    imgthr = cv2.erode(imgDiala,kernel,iterations=1)

    return imgthr

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxa = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(imgcont, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxa and len(approx) == 4:
                biggest = approx
                maxa = area

    cv2.drawContours(imgcont, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder(mypts):
    mypts = mypts.reshape((4,2))
    myptsnew = np.zeros((4,1,2),np.int32)
    add = mypts.sum(1)
    #print("add",add)

    myptsnew[0] = mypts[np.argmin(add)]
    myptsnew[3] = mypts[np.argmax(add)]
    #print("New",myptsnew)
    diff = np.diff(mypts,axis=1)
    myptsnew[1]=mypts[np.argmin(diff)]
    myptsnew[2] = mypts[np.argmax(diff)]
    return myptsnew


def getwrap(img,biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [wi, 0], [0, hi], [wi, hi]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgop = cv2.warpPerspective(img, matrix, (wi, hi))

    imgcropped = imgop[20:imgop.shape[0]-20,20:imgop.shape[1]-20]
    imgcropped = cv2.resize(imgcropped,(wi,hi))

    return imgcropped

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success , img = cap.read()
    #img = cv2.imread(r'C:\Users\muska\Downloads\paper.jpg')
    cv2.resize(img,(wi,hi))
    imgcont = img.copy()
    imgthr = preprocessing(img)
    biggest = getContours(imgthr)
    if biggest.size!=0:
        imgwraped = getwrap(img,biggest)

        #imgarr = ([img,imgthr],[imgcont,imgwraped])
        imgarr = ([imgcont, imgwraped])
        cv2.imshow("Wraped", imgwraped)
    else:
        #imgarr = ([img, imgthr], [img, img])
        imgarr = ([img, imgcont])

    stackedImages = stackImages(0.1,imgarr)

    cv2.imshow("Stacked",stackedImages)
    #cv2.imshow("Wraped",imgwraped)
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break