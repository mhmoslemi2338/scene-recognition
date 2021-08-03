import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage
import copy


def augment_func():
    
    train_path=os.path.join(os.path.abspath("Data"),"Train")
    categories=os.listdir(train_path)
    
    augment =ImageDataGenerator()
        
    for cnt,row in enumerate(categories):
        img_cat=[]
        print(row)
    
        _data_=augment.flow_from_directory(train_path, classes=[row]);
        for i in range(len(_data_)):
            _batch_=_data_[i][0]
            for j,img in enumerate(_batch_):

                (h,w,_)=np.shape(img)

                #### 4 type perspective 
                pt1_1=np.float32([[int(w/14),int(h/14)],[w-int(w/14),int(h/14)],[0,h],[w,h]]) # source
                pt1_2=np.float32([[0,0],[w,0],[int(w/14),h-int(h/14)],[w-int(w/14),h-int(h/14)]]) # source
                pt1_3=np.float32([[int(w/14),int(h/14)],[w,0],[int(w/14),h-int(h/14)],[w,h]]) # source
                pt1_4=np.float32([[0,0],[w-int(w/14),int(h/14)],[0,h],[w-int(w/14),h-int(h/14)]]) # source 
                pt2=np.float32([[0,0],[w,0],[0,h],[w,h]]) # dest

                M=[cv2.getPerspectiveTransform(pt1_1,pt2)]
                M.append(cv2.getPerspectiveTransform(pt1_2,pt2))
                M.append(cv2.getPerspectiveTransform(pt1_3,pt2))
                M.append(cv2.getPerspectiveTransform(pt1_4,pt2))
                
                for m_ in M:
                    img2=copy.deepcopy(img)
                    img2=cv2.warpPerspective(img2,m_,(w,h))
                    img_cat.append(img2)


                #### rotate +-10 degree 
                center=tuple(np.array([h,w])/2)
                rot_mat = [cv2.getRotationMatrix2D(center,10,1.0)]
                rot_mat.append(cv2.getRotationMatrix2D(center,-10,1.0))
                for rot_ in rot_mat:
                    img2=copy.deepcopy(img)
                    img2 = cv2.warpAffine(img2, rot_, (w,h))
                    img_cat.append(img2)
                

                #### flip Horizontally and vertically
                img2=copy.deepcopy(img)
                img3=copy.deepcopy(img)
                img2=cv2.flip(img2,1)
                img3=cv2.flip(img3,0)
                img_cat.append(img2)
                img_cat.append(img3)


                #### draw black box on image
                img2=copy.deepcopy(img)
                box=np.zeros((20,20,3),np.float32) #h,w
                sampl_h = np.uint8(np.random.uniform(low=h/10, high=h-h/10, size=(2,)))
                sampl_w = np.uint8(np.random.uniform(low=w/10, high=w-w/10, size=(2,)))

                for i in sampl_h:
                    for j in sampl_w:     
                        img2[i:i+20,j:j+20]=box # h,w
                img_cat.append(img2)


                #### color map
                img2=copy.deepcopy(img)
                img2 = cv2.cvtColor(np.uint8(img2), cv2.COLOR_BGR2GRAY)
                img2 = cv2.applyColorMap(img2, cv2.COLORMAP_BONE)
                img_cat.append(img2)

                ### 15% zoom-in
                img2=copy.deepcopy(img)
                M = np.array([
                    [1.15, 0, -w*0.075],
                    [0, 1.15, -h*0.075]], dtype=np.float32)
                img2 = cv2.warpAffine(img2, M, (h, w))
                img_cat.append(img2)

                ### 20% zoom-out
                img2=copy.deepcopy(img)
                M = np.array([
                    [0.8, 0, w*0.1],
                    [0, 0.8, h*0.1]], dtype=np.float32)
                img2 = cv2.warpAffine(img2, M, (h, w))
                img_cat.append(img2)


                #### add  random gaussina noise 
                img2=copy.deepcopy(img)
                img2 = (skimage.util.random_noise(np.uint8(img2), mode='gaussian', var=0.007))*255
                img_cat.append(img2)

                #### add  random poison noise  ans salt&pepper
                img2=copy.deepcopy(img)
                img2=(skimage.util.random_noise(skimage.util.random_noise(np.uint8(img2), mode='speckle'), mode='s&p',amount =0.02))*255
                img_cat.append(img2)

        for i,img in enumerate(img_cat):
            name=os.path.join(os.path.join(train_path,row),'aug_'+row+str(i)+'.jpg')
            cv2.imwrite(name,img)
        if cnt==len(categories):
            break
                
    f = open(os.path.join(train_path,"done.txt"), "a")
    f.close()



