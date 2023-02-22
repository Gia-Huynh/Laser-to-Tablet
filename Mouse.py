import mouse
from threading import Thread
#https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
import numpy as np
import cv2
import queue
import time #for benchmarking
from win32api import GetSystemMetrics
unwarp_size = 0. #temporally, check the real value in the middle section
border_size = 0 #temporal
rotate = 0
flip = 1
load_old_perspective = 1

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
    #Half (1 or 2):
    #Reduce image dimension WxH by half -> (W/2) * (H/2) to decrease BFS time
    half = 1
    if img.shape [0] * img.shape [1] > 800 * 800:
        half = 4
        img = img.reshape ((img.shape[0]//half, half, img.shape [1]//half, half)).max(axis=(1,3))
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
    #######################################################################
    #
    #Return to this later please, add an option to specify play area scale
    #
    #######################################################################
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
    """print (ans)
    print ("Chosen value: ")
    print (x_1*half+border_size,' ',y_1*half+border_size)
    print (x_2*half+border_size,' ',y_2*half-border_size)
    print (x_3*half-border_size,' ',y_3*half+border_size)
    print (x_4*half-border_size,' ',y_4*half-border_size)
    return x_1*half+border_size, y_1*half+border_size, x_2*half+border_size, y_2*half-border_size,x_3*half-border_size, y_3*half+border_size, x_4*half-border_size, y_4*half-border_size
    """
    return x_1*half, y_1*half, x_2*half, y_2*half,x_3*half, y_3*half, x_4*half, y_4*half
class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        
        print('Default webcam resolution: ',int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        self.stream.set(cv2.CAP_PROP_FPS, 60) #This line is a joke, my 30fps webcam sucks.
        print('New webcam resolution: ',int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("Your webcam's frame per second: {0}".format(self.stream.get(cv2.CAP_PROP_FPS)))
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self
    def start_show (self, unwarp_matrix, unwarp_size):
        Thread(target=self.show, args=(unwarp_matrix, unwarp_size)).start()
        return self
    def show(self, unwarp_matrix, unwarp_size):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                self.stream.release()
            else:
                cv2.imshow ("OG frame", self.frame)
                cv2.imshow ("Unwarped", cv2.warpPerspective(self.frame, unwarp_matrix, (unwarp_size, unwarp_size)))
                cv2.waitKey(1)
                
    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                self.stream.release()
            else:
                if (flip == 1):
                    self.frame = cv2.flip(self.stream.read()[1], -1)
                else:
                    (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def get_perspective_matrix ():
    print ("Getting Perspective Matrix")
    ans = np.zeros (1)
    i = -1
    while(True):
        frame = video_getter.frame
        #frame = cv2.flip(frame, -1)

        blured_gray_frame = cv2.blur(cv2.cvtColor(cv2.blur(frame, (2, 2)), cv2.COLOR_BGR2GRAY), (2, 2))
        if ((i+1) < num_mean_frame) and (ans.shape == (np.zeros (1)).shape) :
            i = (i+1)
            old_frame [i] = blured_gray_frame
            mean_frame = np.mean (old_frame, axis = 0, dtype = None). astype(np.uint8)
            frame_mean_normalized = cv2.normalize(mean_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            if (i+1) < num_mean_frame:
                continue
            #print ("done taking base frame")
            ans, half = bfs(threshold_visualize (frame_mean_normalized))
            #dunno wtf is this lexsort, but it worked
            #https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
            ans = ans[np.lexsort(np.transpose(ans)[::-1])][::-1]
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = box_anchor_detector (ans, half)
            warped = np.float32([[y_1, x_1], [y_2, x_2], [y_3, x_3], [y_4, x_4]])
            warp_target = np.float32 ([[0, 0], [unwarp_size, 0], [0, unwarp_size], [unwarp_size, unwarp_size]])
            unwarp_matrix = cv2.getPerspectiveTransform(warped, warp_target)  
        else:
            print ("Done")
            return unwarp_matrix
            break

screen_width = float(GetSystemMetrics(0))
screen_height = float(GetSystemMetrics(1))
area_scale = 0.75
area_scale = 1/area_scale
unwarp_size = 1000.
border_size = 0
threshold_grayscale = 210
video_getter = VideoGet().start()

num_mean_frame = 30
video_height = video_getter.height
video_width = video_getter.width
old_frame = np.zeros ((num_mean_frame, video_height, video_width), dtype = np.uint8)
old_frame_normalized = np.zeros ((num_mean_frame, video_height, video_width), dtype = np.uint8)

if (load_old_perspective == 1):
    print ("Loading perspective matrix from file")
    unwarp_matrix = np.load('data.npy')
else:
    unwarp_matrix = get_perspective_matrix()
print ("S T A R T")
#frame_unwarped = cv2.warpPerspective(frame, unwarp_matrix, (int(unwarp_size), int(unwarp_size)))


x_scale = screen_width * area_scale / unwarp_size
y_scale = screen_height * area_scale / unwarp_size

x_constant = 0.5 * (screen_width - screen_width * area_scale)
y_constant = 0.5 * (screen_height - screen_height * area_scale)
unwarp_size = int (unwarp_size)
i=0
video_getter.start_show(unwarp_matrix, unwarp_size)
Time = time.time()
#area_scale = 1./area_scale

try:
    M_old = cv2.moments(
            np.rot90(
                cv2.threshold(
                    cv2.cvtColor(
                        cv2.warpPerspective(video_getter.frame, unwarp_matrix, (unwarp_size, unwarp_size))
                        , cv2.COLOR_BGR2GRAY)
                    ,threshold_grayscale,255,cv2.THRESH_BINARY)[1]
                ,rotate)
            )
    while True:
        frame = video_getter.frame
        M = cv2.moments(
            np.rot90(
                cv2.threshold(
                    cv2.cvtColor(
                        cv2.warpPerspective(frame, unwarp_matrix, (unwarp_size, unwarp_size))
                        , cv2.COLOR_BGR2GRAY)
                    ,threshold_grayscale,255,cv2.THRESH_BINARY)[1]
                ,rotate)
            )
        if ((M["m00"]>0.00001)
        and (np.linalg.norm(((M["m10"] / (M["m00"]+0.00001)) - (M_old["m10"] / (M_old["m00"]+0.00001)), (M["m01"] / (M["m00"]+0.00001)- (M_old["m01"] / (M_old["m00"]+0.00001)))))
             > 0.1)):
                mouse.move (((M["m10"] / M["m00"]) * x_scale  + x_constant),
                            ((M["m01"] / M["m00"]) * y_scale  + y_constant))
        M_old = M
        i+=1
except KeyboardInterrupt:
    pass
    
delta_time = time.time() - Time
print ("Average time for a frame: ", delta_time/float(i))
print ("vs 1/30 seconds: ", (delta_time/float(i))/(1./30.))

try:
    np.save('data.npy', unwarp_matrix) # save
    print ("Saved perspective matrix")
except Exception as e:
    print (e.__doc__)
    print (e.message)  
video_getter.stop()
