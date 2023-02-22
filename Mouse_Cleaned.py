import mouse
import _thread
import numpy as np
import cv2
import queue
import time #for benchmarking
unwarp_size = 0. #temporal
border_size = 0 #temporal
rorate = 180 #0, 90, 180, 270

def clahee1(img, normalize, clipLimit=1.5, tileGridSize=(8,8)):
    if (normalize == 1):
        img2 = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    else:
        img2 = np.copy (img)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    img2 [:,:] = clahe.apply(img2[:,:])
    return img2

def clahee(img, normalize, clipLimit=1.5, tileGridSize=(8,8)):
    if img.ndim == 2:
        return clahee1(img, normalize, clipLimit, tileGridSize)
    if (normalize == 1):
        img2 = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    img2 [:,:,0] = clahe.apply(img2[:,:,0])
    img2 [:,:,1] = clahe.apply(img2[:,:,1])
    img2 [:,:,2] = clahe.apply(img2[:,:,2])
    return img2

def threshold_visualize (img, convert = 0):
    if convert == 1:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy (img)
    thresh_inv = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)[1]
    blur = cv2.GaussianBlur(thresh_inv,(9, 9),0)
    #thresh_inv = blur
    thresh_inv = cv2.threshold(blur,25,255,cv2.THRESH_BINARY_INV)[1]
    return thresh_inv

def bfs (img):
    half = 1
    if img.shape [0] * img.shape [1] > 800 * 800:
        half = 2
        img = img.reshape ((img.shape[0]//2, 2, img.shape [1]//2, 2)).max(axis=(1,3))
    #help i don't know how to bfs fast in python, jk lmao we only need to run this 1 time so it doesn't matter much
    visited = np.zeros (img.shape, dtype = np.uint8)
    q = queue.Queue()
    count = -1
    #1: Top Left
    #2: Top Right
    #3: Bottom Left
    #4: Bottom Right
    #[area, X1, Y1, X2, Y2, X3, Y3, X4, Y4]
    #[area, maxX, maxY, minX, minY]
    count_ghey = np.zeros ((1000, 9), dtype = np.int32)
    count_ghey.fill (-1)
    count_ghey [:,1].fill(9999)
    count_ghey [:,2].fill(9999)
    count_ghey [:,3].fill(9999)
    count_ghey [:,6].fill(9999)
    
    startx = 0
    starty = 0
    count_random = 0
    epsilon = int(min (img.shape[0], img.shape[1]) * 5. /100.)
    while (img[startx, starty] != 255 ):
        startx = np.random.randint(img.shape[0])
        starty = np.random.randint(img.shape[1])
        count_random += 1
        if count_random == 1000:
            return count_ghey, half
    while True:
        count += 1
        path = [startx, starty]
        q.put(path)
        while not q.empty():
            path = q.get()
            #print (path)
            x = path [0]
            y = path [1]
            visited [x, y] = 255
            #increase area by 1
            count_ghey [count, 0] += 1
            
            #check for bounding box co-ordinate
            #We need 4 points for unwarping process
            #if (((count_ghey [count, 1] - x) / (count_ghey [count, 2] - y +0.01) < 0.75) and
            if ((((count_ghey [count, 2] - y) / (count_ghey [count, 1] - x + 0.0001) > -0.7) and ((count_ghey [count, 1] - x) > 0))
                or (((count_ghey [count, 2] - y) / (count_ghey [count, 1] - x + 0.0001) < -2) and ((count_ghey [count, 1] - x) < 0))):
                    count_ghey[count, 2] = y;
                    count_ghey[count, 1] = x
                    
            if ((((count_ghey [count, 4] - y) / (count_ghey [count, 3] - x + 0.0001) < 0.7) and ((count_ghey [count, 3] - x) > 0))
                or (((count_ghey [count, 4] - y) / (count_ghey [count, 3] - x + 0.0001) > 2) and ((count_ghey [count, 3] - x) < 0))):
                    count_ghey[count, 4] = y;
                    count_ghey[count, 3] = x
                    
            if ((((count_ghey [count, 6] - y) / (count_ghey [count, 5] - x + 0.0001) < 0.7) and ((count_ghey [count, 5] - x) < 0))
                or (((count_ghey [count, 6] - y) / (count_ghey [count, 5] - x + 0.0001) > 2) and ((count_ghey [count, 3] - x) > 0))):
                    count_ghey[count, 6] = y;
                    count_ghey[count, 5] = x
                    
            if ((((count_ghey [count, 8] - y) / (count_ghey [count, 7] - x + 0.0001) > -0.7) and ((count_ghey [count, 7] - x) < 0))
                or (((count_ghey [count, 8] - y) / (count_ghey [count, 7] - x + 0.0001) < -2) and ((count_ghey [count, 7] - x) > 0))):
                    count_ghey[count, 8] = y;
                    count_ghey[count, 7] = x
                    
    
            for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                try:
                    if (x2<0) or (y2<0):
                        continue
                    if ((img[x2, y2] == 255) and (visited [x2, y2] == 0)):
                        q.put ([x2, y2])
                        visited [x2, y2] = 255
                except IndexError:
                    continue
        count_random = 0        
        while (img[startx, starty] != 255 ) or (visited [startx, starty] != 0):
            startx = np.random.randint(img.shape[0])
            starty = np.random.randint(img.shape[1])
            count_random += 1
            if count_random == 1000:
                return count_ghey, half
def box_anchor_detector (ans, half):
    candidate_0 = ans [0]
    candidate_1 = ans [1]
    if (candidate_0[1] < candidate_1[1]):
        x_1 = candidate_1[1]
        y_1 = candidate_1[2]
        x_2 = candidate_1[3]
        y_2 = candidate_1[4]
        x_3 = candidate_1[5]
        y_3 = candidate_1[6]
        x_4 = candidate_1[7]
        y_4 = candidate_1[8]
    else:
        x_1 = candidate_0[1]
        y_1 = candidate_0[2]
        x_2 = candidate_0[3]
        y_2 = candidate_0[4]
        x_3 = candidate_0[5]
        y_3 = candidate_0[6]
        x_4 = candidate_0[7]
        y_4 = candidate_0[8]
    print (ans)
    print ("Chosen value: ")
    print (x_1*half+border_size,' ',y_1*half+border_size)
    print (x_2*half+border_size,' ',y_2*half-border_size)
    print (x_3*half-border_size,' ',y_3*half+border_size)
    print (x_4*half-border_size,' ',y_4*half-border_size)
    return x_1*half+border_size, y_1*half+border_size, x_2*half+border_size, y_2*half-border_size,x_3*half-border_size, y_3*half+border_size, x_4*half-border_size, y_4*half-border_size

def grey_filter (frame):
    M = cv2.moments(cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),200,255,cv2.THRESH_BINARY)[1])
    if (M["m00"]>0.00001):
        mouse.move ((M["m10"] / M["m00"])/unwarp_size*1920,(M["m01"] / M["m00"])/unwarp_size*1080)
def webcam_setup ():
    
i=-1
unwarp_size = 1000.
border_size = 25
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Old webcam resolution: ',width,height)

HIGH_VALUE = 10000
WIDTH = HIGH_VALUE
HEIGHT = HIGH_VALUE

#Dunno what this shit does, increase webcam res to max limit?
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('New webcam resolution: ',width,height)

fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
num_mean_frame = 30
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH  ))
old_frame = np.zeros ((num_mean_frame, video_height, video_width), dtype = np.uint8)
old_frame_normalized = np.zeros ((num_mean_frame, video_height, video_width), dtype = np.uint8)
ans = np.zeros (1)  
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.flip(frame, -1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_normalized = cv2.normalize(frame_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray_blured_frame = cv2.cvtColor(cv2.blur(frame, (5, 5)), cv2.COLOR_BGR2GRAY)
    blured_gray_frame = cv2.blur(cv2.cvtColor(cv2.blur(frame, (2, 2)), cv2.COLOR_BGR2GRAY), (2, 2))
    if ((i+1) < num_mean_frame) and (ans.shape == (np.zeros (1)).shape) :
        i = (i+1)
        old_frame [i] = blured_gray_frame
        mean_frame = np.mean (old_frame, axis = 0, dtype = None). astype(np.uint8)
        frame_mean_normalized = cv2.normalize(mean_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        frame_mean_normalized_clahe = clahee1 (frame_mean_normalized, 0, clipLimit=20, tileGridSize=(4, 4))
        if (i+1) < num_mean_frame:
            continue
        print ("done taking base frame")
        ans, half = bfs(threshold_visualize (frame_mean_normalized))
        ans = ans[np.lexsort(np.transpose(ans)[::-1])][::-1]
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = box_anchor_detector (ans, half)
        warped = np.float32([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]])
        warp_target = np.float32 ([[0, 0], [unwarp_size, 0], [0, unwarp_size], [unwarp_size, unwarp_size]])
        unwarp_matrix = cv2.getPerspectiveTransform(warped, warp_target)  
    else:
        break

#frame_unwarped = cv2.warpPerspective(frame, unwarp_matrix, (int(unwarp_size), int(unwarp_size)))    
print ("Let's go boiz")
x_scale = float(unwarp_size/1920)
y_scale = float(unwarp_size/1080)
unwarp_size = int (unwarp_size)
i=0
Time = time.time()
print ("Time Old:", Time)
try:
    while True:
        #frame = cap.read()[1]
        M = cv2.moments(cv2.threshold(cv2.cvtColor(cv2.warpPerspective(frame, unwarp_matrix, (unwarp_size, unwarp_size)), cv2.COLOR_BGR2GRAY),200,255,cv2.THRESH_BINARY)[1])
        if (M["m00"]>0.00001):
            mouse.move ((M["m10"] / M["m00"])/x_scale,(M["m01"] / M["m00"])/y_scale)
        else: #benchmarking time
            M["m10"]
            M["m00"]
            M["m01"]
            M["m00"]
            mouse.move ((100. / 1000.)/x_scale,(100. / 1000.)/y_scale)
        i+=1
except KeyboardInterrupt:
    pass
print ("Time New:", time.time())
delta_time = time.time() - Time
#print ("Time (vs 1/30 seconds): ", delta_time/(1./30.))
print ("Average time: ", delta_time/float(i))
print ("vs 1/30 seconds: ", (delta_time/float(i))/(1./30.))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#ans = bfs(threshold_visualize (frame_mean_normalized))
#dunno wtf is this sort, but it worked
#https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
#ans = ans[np.lexsort(np.transpose(ans)[::-1])][::-1]
    

