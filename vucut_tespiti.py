import cv2
#CTRL basılı tutarak cv2 gir ve xml kütüphanelerini al
#Açılan dosyanın üzerine sağ click ile gir ve dosyaları göster kısmına git.
#Data içerisinde haar_cascade var

import numpy as np

cam=cv2.VideoCapture("test.avi")

tum_vucut=cv2.CascadeClassifier("haarcascade_fullbody.xml")
alt_vucut=cv2.CascadeClassifier("haarcascade_lowerbody.xml")
ust_vucut=cv2.CascadeClassifier("haarcascade_upperbody.xml")

#Resmi oku
ret,resim=cam.read()

detect=np.zeros([resim.shape[0],resim.shape[1],3],np.uint8)



def nothing(x):
    pass

#Trackbar
cv2.namedWindow("vucut",cv2.WINDOW_NORMAL)
cv2.createTrackbar("one_param","vucut",0,100,nothing)
cv2.createTrackbar("two_param","vucut",0,100,nothing)
cv2.createTrackbar("switch","vucut",0,1,nothing)


while cam.isOpened():
    detect[:]=0
    if cv2.getTrackbarPos("switch","vucut")==1:
        cv2.waitKey(1)
        continue
    
    ret,resim=cam.read()
    
    if not ret:
        print("Vucut Bulunuyor..")
        break
    
    resim_gri=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)
    
    one_param=cv2.getTrackbarPos("one_param","vucut")/10+1.01
    two_param=cv2.getTrackbarPos("two_param","vucut")+1
    
    print("Ilk parametre: {},Ikınci parametre: {}".format(one_param,two_param))
    
    vucutlar=tum_vucut.detectMultiScale(resim_gri,one_param,two_param,minSize=(30,30),maxSize=(100,100))
    
    alt_vucutlar=alt_vucut.detectMultiScale(resim_gri,one_param,two_param,minSize=(30,30),maxSize=(100,100))
     
    ust_vucutlar=ust_vucut.detectMultiScale(resim_gri,one_param,two_param,minSize=(30,30),maxSize=(100,100))
    
    for x,y,w,h in vucutlar:
        detect[y:y+h,x:x+w]=resim[y:y+h,x:x+w]
        cv2.rectangle(resim,(x,y),(x+w,y+h),(255,0,0),2)
        resim[y:y+h,x:x+w,0]=255#Alanın içini boyama
        
    for x,y,w,h in alt_vucutlar:
        cv2.rectangle(resim,(x,y),(x+w,y+h),(0,255,0),2)
        resim[y:y+h,x:x+w,0]=255#Alanın içini boyama
        
    for x,y,w,h in ust_vucutlar:
        cv2.rectangle(resim,(x,y),(x+w,y+h),(0,0,255),2)
        resim[y:y+h,x:x+w,0]=255#Alanın içini boyama
        
    #Tüm resimde arama
        
    
    cv2.imshow("vucut",resim)
    cv2.imshow("detect",detect)
    
    
    if cv2.waitKey(5)==ord("q"):
        print("Cikis")
        break
    
cv2.destroyAllWindows()
cam.release()

    










