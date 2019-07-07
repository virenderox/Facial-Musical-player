from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import vlc
import random as r
import time
import music


def play_music(s = 0):
    instance = vlc.Instance()
    media = instance.media_new(s)
    player = instance.media_player_new()
    player.set_media(media)
    return(player)

def facial_model():
    v = cv2.VideoCapture(0)
    model = load_model('_mini_XCEPTION.106-0.65.hdf5',compile = False)
    fd = cv2.CascadeClassifier(r'C:\Users\Virender Pal Singh\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    em = ['angry','disgust','fear','happy','sad','surprised','neutral']
    music_path = music.music_path()
    while(1):
        r,i = v.read()
        j = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) # change 
        f = fd.detectMultiScale(j,1.2,5)
        if len(f) > 0:
            
            [x,y,w,h] = f[0]
            cv2.rectangle(i,(x,y),(x+w,y+h),(0,0,255),5)
            roi = j[y:y+h,x:x+w]
            roi = cv2.resize(roi,(48,48))
            roi = roi.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)
            p = list(model.predict(roi)[0])

            for j in em:
                if (em[p.index(max(p))]) == j:
                    #a = r.randint(0,len(music_path[i]-1))
                    player = play_music(music_path[j][0])
                    player.play()
                    time.sleep(5000)
##        else:
##            player = play_music()
##            player.pause()
##            
        cv2.imshow('image', i)
        k = cv2.waitKey(5)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break

facial_model()
