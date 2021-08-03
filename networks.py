from tensorflow.keras import layers , models ,initializers
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import torch
import torchvision


def network1(input_shape):


    X_input=layers.Input(input_shape)
    X = layers.Conv2D(96,(11,11),strides = 4,name="conv0",kernel_initializer=initializers.GlorotUniform())(X_input)
    X = layers.BatchNormalization(axis = 3 , name = "batchnormal0")(X)
    X = layers.Activation('relu')(X)

    X = layers.MaxPooling2D((4,4),strides = 4,name = 'maxpool_0')(X)
    X = layers.Dropout(0.5)(X)
    

    X = layers.Flatten()(X)

    X = layers.Dense(4096, activation = 'relu', name = 'full_connect0',kernel_initializer=initializers.GlorotUniform())(X) 
    X = layers.Dropout(0.5)(X)
    X = layers.Dense(15,activation='softmax',name = 'full_connect1',kernel_initializer=initializers.GlorotUniform())(X)

    
    model = models.Model(inputs = X_input, outputs = X, name='network1')
    opt=SGD(learning_rate=0.001 , momentum=0.97)
    m1=tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_1_accuracy", dtype=None)
    m2=tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy", dtype=None)
    
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=[m1,m2])

    return model

def network2(input_shape):

    X_input = layers.Input(input_shape)
    
    X = layers.Conv2D(96,(11,11),strides = 4,name="conv0",kernel_initializer=initializers.GlorotUniform())(X_input)
    X = layers.BatchNormalization(axis = 3 , name = "batchnormal0")(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3,3),strides = 2,name = 'maxpool_0')(X)
    X = layers.Dropout(0.5)(X)
    
    X = layers.Conv2D(256,(5,5),padding = 'same' , name = 'conv1',kernel_initializer=initializers.GlorotUniform())(X)
    X = layers.BatchNormalization(axis = 3 ,name='batchnormal1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3,3),strides = 2,name = 'maxpool_1')(X)
    X = layers.Dropout(0.5)(X)

    X = layers.Conv2D(256, (3,3) , padding = 'same' , name='conv2',kernel_initializer=initializers.GlorotUniform())(X)
    X = layers.BatchNormalization(axis = 3, name = 'batchnormal2')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2),strides = 2,name = 'maxpool_2')(X)
    X = layers.Dropout(0.5)(X)
  
    X = layers.Flatten()(X)
    X = layers.Dense(4096, activation = 'relu', name = 'full_connect0',kernel_initializer=initializers.GlorotUniform())(X) 
    X = layers.Dropout(0.5)(X)

    X = layers.Dense(4096, activation = 'relu', name = 'full_connect1',kernel_initializer=initializers.GlorotUniform())(X) 
    X = layers.Dropout(0.5)(X)
    
    X = layers.Dense(15,activation='softmax',name = 'full_connect2',kernel_initializer=initializers.GlorotUniform())(X)

    model = models.Model(inputs = X_input, outputs = X, name='network1')
    opt=SGD(learning_rate=0.001 , momentum=0.97)
    m1=tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_1_accuracy", dtype=None)
    m2=tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy", dtype=None)
    
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=[m1,m2])
    return model







def network3(input_shape):

      
    X_input = layers.Input(input_shape)
    
    X = layers.Conv2D(96,(11,11),strides = 4,name="conv0",kernel_initializer=initializers.GlorotUniform())(X_input)
    X = layers.BatchNormalization(axis = 3 , name = "batchnormal0")(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3,3),strides = 2,name = 'maxpool_0')(X)
    X = layers.Dropout(0.5)(X)
    
    X = layers.Conv2D(256,(5,5),padding = 'same' , name = 'conv1',kernel_initializer=initializers.GlorotUniform())(X)
    X = layers.BatchNormalization(axis = 3 ,name='batchnormal1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3,3),strides = 2,name = 'maxpool_1')(X)
    X = layers.Dropout(0.5)(X)

    X = layers.Conv2D(384, (3,3) , padding = 'same' , name='conv2',kernel_initializer=initializers.GlorotUniform())(X)
    X = layers.BatchNormalization(axis = 3, name = 'batchnormal2')(X)
    X = layers.Activation('relu')(X)
    
    
    X = layers.Conv2D(384, (3,3) , padding = 'same' , name='conv3',kernel_initializer=initializers.GlorotUniform())(X)
    X = layers.BatchNormalization(axis = 3, name = 'batchnormal3')(X)
    X = layers.Activation('relu')(X)


    X = layers.Conv2D(256, (3,3) , padding = 'same' , name='conv4',kernel_initializer=initializers.GlorotUniform())(X)
    X = layers.BatchNormalization(axis = 3, name = 'batchnormal4')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3,3),strides = 2,name = 'max4')(X)
    X = layers.Dropout(0.5)(X)


    X = layers.Flatten()(X)
    X = layers.Dense(4096, activation = 'relu', name = 'full_connect0',kernel_initializer=initializers.GlorotUniform())(X) 
    X = layers.Dropout(0.5)(X)

    X = layers.Dense(4096, activation = 'relu', name = 'full_connect1',kernel_initializer=initializers.GlorotUniform())(X) 
    X = layers.Dropout(0.5)(X)
    
    X = layers.Dense(15,activation='softmax',name = 'full_connect2',kernel_initializer=initializers.GlorotUniform())(X)

    model = models.Model(inputs = X_input, outputs = X, name='network1')
    opt=SGD(learning_rate=0.001 , momentum=0.97)
    m1=tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_1_accuracy", dtype=None)
    m2=tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy", dtype=None)
    
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=[m1,m2])
    return model





def network4():
    
    model = torchvision.models.alexnet(pretrained=True)

    seq1 = list(model.features.children())
    seq2 = list(model.classifier.children())[0:-1]
    last_layer=torch.nn.Linear(in_features=4096, out_features=15, bias=True)
    torch.nn.init.trunc_normal_(last_layer.bias)
    torch.nn.init.kaiming_uniform_(last_layer.weight)

    seq2.append(last_layer)
    seq1.append(torch.nn.Flatten())
    for row in seq2:
      seq1.append(row)
    model=torch.nn.Sequential(*seq1)


    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model[20].parameters() :
        parameter.requires_grad = True

    return model



def network5():
    
    model = torchvision.models.alexnet(pretrained=True)

    seq1 = list(model.features.children())
    seq2 = list(model.classifier.children())[0:-1]
    last_layer=torch.nn.Linear(in_features=4096, out_features=15, bias=True)
    torch.nn.init.trunc_normal_(last_layer.bias)
    torch.nn.init.kaiming_uniform_(last_layer.weight)

    seq2.append(last_layer)
    seq1.append(torch.nn.Flatten())
    for row in seq2:
      seq1.append(row)
    model=torch.nn.Sequential(*seq1)

    return model

