


import matplotlib.pyplot as plt
import os , pickle
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import torch

def saveVar(myvar,name,path):
  name=os.path.join(path,name+'.pckl')
  f = open(name, 'wb')
  pickle.dump(myvar, f)
  f.close()
  return


def readvar(name,path):
  name=os.path.join(path,name+'.pckl')
  f = open(name, 'rb')
  myvar = pickle.load(f)
  f.close()
  return myvar


def fit_model(epoch,model,Train,Test,path):
    accuracy1,accuracy5,loss=[],[],[]
    accuracy1_test,accuracy5_test,loss_test=[],[],[]
    
    checkpoint_path = path+"/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    
    for i in range(epoch):
        print("Epoch "+str(i+1)+'/'+str(epoch))

        model.fit(Train,epochs=1,callbacks=[cp_callback])
        
        accuracy1.append(model.history.history['top_1_accuracy'][0])
        accuracy5.append(model.history.history['top_5_accuracy'][0])
        loss.append(model.history.history['loss'][0])
        train_hist={'loss':loss,'accuracy1':accuracy1,'accuracy5':accuracy5}
        
        predict=model.evaluate(Test)
        loss_test.append(predict[0])
        accuracy1_test.append(predict[1])
        accuracy5_test.append(predict[2])
        test_hist={'loss':loss_test,'accuracy1':accuracy1_test,'accuracy5':accuracy5_test}

        saveVar(train_hist,'train_hist',checkpoint_dir)
        saveVar(test_hist,'test_hist',checkpoint_dir)

    return [model ,train_hist ,test_hist]




def fit_model_torch(epochs,model,optimizer,Train,Test,path):
   
    
    criterion = torch.nn.CrossEntropyLoss()
    accuracy1,accuracy5,loss_train=[],[],[]
    accuracy1_test,accuracy5_test,loss_test=[],[],[]
    for epoch in range(epochs):
        epoch_correct1,epoch_correct5,epoch_loss=0,0,0
        
        for i, data in enumerate(Train,0):
            (images, labels)=data
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            _, pred5 = outputs.data.topk(5, dim=1)
            _, pred1 = outputs.data.topk(1, dim=1)

            epoch_correct1+=(pred1.t() == labels).sum().item()
            for j,row in enumerate(labels):
                if (row.item() in pred5[j]):
                    epoch_correct5+=1
            
            
            if i%50==49:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f' %(epoch+1,epochs,i+1,int(np.ceil(len(Train.dataset)/64)),epoch_loss/(i+1)))
        

        loss_train.append(epoch_loss/(i+1))
        accuracy1.append(epoch_correct1/len(Train.dataset))
        accuracy5.append(epoch_correct5/len(Train.dataset))
        train_hist={'loss':loss_train,'accuracy1':accuracy1,'accuracy5':accuracy5}
        
        
        epoch_correct1,epoch_correct5,epoch_loss=0,0,0
        model.eval()  
        criterion = torch.nn.CrossEntropyLoss()
        for i, (images, labels) in enumerate(Test):
            outputs = model(images)
            loss = criterion(outputs,labels)
            epoch_loss+=loss.item()

            _, pred5 = outputs.data.topk(5, dim=1)
            _, pred1 = outputs.data.topk(1, dim=1)

            epoch_correct1+=(pred1.t() == labels).sum().item()
            for j,row in enumerate(labels):
                if (row.item() in pred5[j]):
                    epoch_correct5+=1


        loss_test.append(epoch_loss/(i+1))
        accuracy1_test.append(epoch_correct1/len(Test.dataset))
        accuracy5_test.append(epoch_correct5/len(Test.dataset))
        print('acc_top1 on test : %.4f  , loss : %.6f' % (accuracy1_test[-1],loss_test[-1]))

        test_hist={'loss':loss_test,'accuracy1':accuracy1_test,'accuracy5':accuracy5_test}
        
        saveVar(train_hist,'train_hist',path)
        saveVar(test_hist,'test_hist',path)

    return [model ,train_hist ,test_hist]













def show_result(train_hist,test_hist,num):
    acc1 =train_hist['accuracy1']
    val_acc1=test_hist['accuracy1']
    
    acc5 =train_hist['accuracy5']
    val_acc5=test_hist['accuracy5']
    

    loss = train_hist['loss']
    val_loss=test_hist['loss']

    epochs = range(1, len(acc1) + 1)

    fig=plt.figure(figsize=(27,5))
    plt.subplot(131)
    plt.plot(epochs, acc1, '*--', label='Training acc')
    plt.plot(epochs, val_acc1, '*--', label='Validation acc')
    plt.title('Training and Test accuracy for top1'); plt.legend()
    plt.grid( which='major', color='g', linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel("Acuuracy")
    
    plt.subplot(132)
    plt.plot(epochs, acc5, '*--', label='Training acc')
    plt.plot(epochs, val_acc5, '*--', label='Validation acc')
    plt.title('Training and Test accuracy for top5'); plt.legend()
    plt.grid( which='major', color='g', linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel("Acuuracy")

    plt.subplot(133)
    plt.plot(epochs, loss, '*--', label='Training loss')
    plt.plot(epochs, val_loss, '*--', label='Validation loss')
    plt.title('Training and Test loss'); plt.legend()
    plt.grid( which='major', color='g', linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    fig.savefig(num+'.jpg', dpi=4*fig.dpi)
    plt.close(fig)





