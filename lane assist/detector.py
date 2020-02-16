import matplotlib.pylab as plt
import cv2
import numpy as np
import math
from scipy.ndimage import rotate
import time
from scipy.spatial import distance

##def closest_node(node, nodes):
##    distance = 1000
##    index = 0
##    print(node, nodes)
##    for i in range(len(nodes[0])):
##        dis = math.sqrt((node[0]-nodes[0][i][0])**2(node[1]-nodes[0][i][1])**2)
##        if dis<distance:
##            distance=dis
##            index = i
##    return nodes[0][index]

##Izrezovanje slike na trikotnik
def obmocje_interesa(slika, tocke):
    maska = np.zeros_like(slika)
    #st_kanalov = slika.shape[2]
    ujemajoca_barvna_maska = 255#(,) * st_kanalov
    cv2.fillPoly(maska, tocke, ujemajoca_barvna_maska)
    maskirana_slika = cv2.bitwise_and(slika, maska)
    return maskirana_slika

def narisi_crte(slika, crte):
    slika1 = np.copy(slika)
                #3 channels
    prazna_slika = np.zeros((slika1.shape[0], slika1.shape[1], 3), dtype=np.uint8)
    for crta in crte:
        for x1, y1, x2, y2 in crta:
            cv2.line(prazna_slika, (x1, y1), (x2, y2), (255, 100, 50), thickness=2)

    slika1 = cv2.addWeighted(slika1, 0.6, prazna_slika, 1, 0.0)
    return slika1
##            def HughToLineMask(points, hughLinesPResult, slika):
##                xyHough = []
##                for point in points:
##                    sdist = 10000
##                    cur_xy = []
##                    ##LEVA STRAN
##                    for line in hughLinesPResult[0]:
##                        dist1 = abs(point[0] - line[0]) + abs(point[1] - line[1])
##                        dist2 = abs(point[0] - line[2]) + abs(point[1] - line[3])
##                        if dist1 < sdist:
##                            sdist=dist1
##                            cur_xy.clear()
##                            cur_xy.append(line[0])
##                            cur_xy.append(line[1])
##                        if dist2 < sdist:
##                            sdist=dist2
##                            cur_xy.clear()
##                            cur_xy.append(line[2])
##                            cur_xy.append(line[3])
##                    xyHough.append(cur_xy)
##
##                ##Maska
##                slika1 = np.copy(slika)
##                            #3 channels
##                prazna_slika = np.zeros((slika1.shape[0], slika1.shape[1], 3), dtype=np.uint8)
##                cv2.line(prazna_slika, (xyHough[0][0], xyHough[0][1]), (xyHough[1][0], xyHough[1][1]), (255, 100, 50), thickness=2)
##                cv2.line(prazna_slika, (xyHough[2][0], xyHough[2][1]), (xyHough[3][0], xyHough[3][1]), (255, 100, 50), thickness=2)
##                slika1 = cv2.addWeighted(slika1, 0.6, prazna_slika, 1, 0.0)
##                return slika1  

##PRETVORBA VIDEO V SLIKE
vidcap = cv2.VideoCapture('detection.mp4')
success,image = vidcap.read()
count = 0
alpha = 2 # Contrast control (1.0-3.0)
beta = -140 # Brightness control (0-100)
img_array = []
while success:
##    start_time = time.time()
    if count%2==0:
        try:
            image =  rotate(image, -90)
            image = image[500:1000, 0:720]
            slika_ceste = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            slika_ceste = cv2.convertScaleAbs(slika_ceste, alpha=alpha, beta=beta)
##            print("01")
##            with np.nditer(slika_ceste, op_flags=['readwrite']) as it:
            for x in range(500):
                for y in range(720):
                    
                    if (slika_ceste[x,y][0]<130) or (slika_ceste[x,y][1]<130) or (slika_ceste[x,y][2]<130):
                        slika_ceste[x,y][0]=0
                        slika_ceste[x,y][1]=0
                        slika_ceste[x,y][2]=0
##            print("03")
##            slika_ceste[slika_ceste < 200] = 0
            plt.axis('off')
##            image.axes.get_xaxis().set_visible(False)
##            image.axes.get_yaxis().set_visible(False)
            ##recognizer=cv2.face.EigenFaceRecognizer_create()
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            visina = slika_ceste.shape[0]
            sirina = slika_ceste.shape[1]


            ##Določevsnje trikotnika / območja
            obmocje_interesa_dolocitev = [
                (0, visina),
                (0, visina-50),
                (sirina/2, visina/2),
                (sirina, visina-50),
                (sirina, visina)
            ]


            

                

            ##gray scaling slike
            siva_slika = cv2.cvtColor(slika_ceste, cv2.COLOR_RGB2GRAY)

            blur = cv2.blur(siva_slika,(4, 4), 0)#cv2.BORDER_DEFAULT)

            ret,thresh4 = cv2.threshold(blur,100,255,cv2.THRESH_TOZERO)
            
            ##Prepoznava po robovih
            canny_slika = cv2.Canny(thresh4, 0, 255)
            ##izrezana slika
            izrezana_slika = obmocje_interesa(canny_slika,
                            np.array([obmocje_interesa_dolocitev], np.int32),)

            crte = cv2.HoughLinesP(izrezana_slika,
                                   rho=6,
                                   theta=np.pi/60,
                                   threshold=180,
                                   lines=np.array([]),
                                   minLineLength=40,
                                   maxLineGap=25)

            slika_s_crtami = narisi_crte(image, crte)

##            print(crte)
            plt.imshow(slika_s_crtami)
            slika = "result/result"+str(count)+".png"
##            plt.savefig(slika, bbox_inches='tight', pad_inches = 0)
##            plt.show()     # save frame as JPEG file
            img_array.append(slika_s_crtami)
        except Exception as x:
            img_array.append(image)
##            print(x)
    success,image = vidcap.read()
    count += 1
##    elapsed_time = time.time() - start_time
##    print(elapsed_time)
    print(count)
    

##slika_ceste = cv2.imread("images/road2.png")
##slika_ceste = cv2.cvtColor(slika_ceste, cv2.COLOR_BGR2RGB)


out = cv2.VideoWriter('project5.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (720, 500))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
