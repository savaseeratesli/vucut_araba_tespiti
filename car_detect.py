import cv2

def main():
    car_cascade=cv2.CascadeClassifier("cars.xml")
    
    cam=cv2.VideoCapture("car1.avi")
    
    while cam.isOpened():
        
        ret,frame=cam.read()
        
        if not ret:
            print("Kapat")
            break
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
        #Histogram ayrıntıları gösterme
        gray=cv2.equalizeHist(gray)
        
        cars=car_cascade.detectMultiScale(gray,1.1,2)
        
        for x,y,w,h in cars:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            frame[y:y+h,x:x+w,2]=255
            
        cv2.imshow("frame",frame)
        
        if cv2.waitKey(33)&0xFF==ord("q"):
            break
    
    cam.release()      
    cv2.destroyAllWindows()

#Dosyanın ismi main ise
#Kod import ediliyorsa farklı isimde olur bu main çalışmaz car,otobus,bisiklet vb...
if __name__=="__main__":
    main()

#Farklı klasörde çalıştırmak için
#import car_detect
#car_detect.main()
#cascade yapısını değiştirerek farklı nesneler bulunabilir.




