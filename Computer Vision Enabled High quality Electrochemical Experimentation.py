import cv2
from audioop import minmax
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from PIL import Image
import glob
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.layers import BatchNormalization
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing import image
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import random
import csv

#DEBUG-----------------------------------------------------------------------------
DEBUG = False

#DEF-------------------------------------------------------------------------------
def square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        result = pil_img
        return result
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), (background_color))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), (background_color))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def pil2opencv(in_image):
    out_image = np.array(in_image, dtype=np.uint8)
    return out_image

def opencv2pil(in_image):
    new_image = in_image.copy()   
    new_image = Image.fromarray(new_image)
    return new_image

def hough(img):
    img_median = cv2.medianBlur(img, 15)

    circles = cv2.HoughCircles(img_median, cv2.HOUGH_GRADIENT, 1.5, 100,
                                param1=150, param2=100, minRadius=50, maxRadius=300)

    circles = np.uint16(np.around(circles))
    img_edges_hough = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        cv2.circle(img_edges_hough,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(img_edges_hough,(i[0],i[1]),2,(0,0,255),3)
        
        HARD_LIMIT_DIM = 288
        biggestCircle = [0,0,0]
        for i in circles[0,:]:
            z = i[2]
            if  (z > biggestCircle[2]) and (z < HARD_LIMIT_DIM):
                biggestCircle = i
        cv2.circle(img_edges_hough,(biggestCircle[0],biggestCircle[1]),biggestCircle[2],(255,0,0),10)
        cv2.circle(img_edges_hough,(biggestCircle[0],biggestCircle[1]),2,(255,0,0),10)
        outerRad = int(biggestCircle[2])
        shape = img_edges_hough.shape[1::-1]
        outerCircle = np.zeros((shape[1],shape[0]),dtype=np.uint8)
        cv2.circle(outerCircle,center=(biggestCircle[0],biggestCircle[1]),radius=outerRad,color=255,thickness=-1)
        ori[outerCircle==0] = [255]
    return ori

def gray_to_rgb(X):
    X_transpose = np.array(X.transpose(0, 1, 2, 3))
    ret = np.empty((X.shape[0], 288, 288, 3), dtype=np.float32)
    ret[:, :, :, 0] = X_transpose[:, :, :, 0]
    ret[:, :, :, 1] = X_transpose[:, :, :, 0]
    ret[:, :, :, 2] = X_transpose[:, :, :, 0]
    return ret.transpose(0, 1, 2, 3)

folder = ['Image']
Class = ['Retry', 'Pass']
image_size_x = 288 
image_size_y = 216
T_value = 2.082677039
ON_error = 0.07
ON_line_upper = (1+ON_error)*T_value
ON_line_lower = (1-ON_error)*T_value

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

slope = np.loadtxt('ORR.csv')
print(slope)

for index, name in enumerate(folder): 
    dir = name
    files = glob.glob(dir + "\*.jpg") 
    print(files)
    
    random_state = 0
    fx_train, fx_test, fy_train, fy_test = train_test_split(files, slope, test_size=0.3, random_state=random_state)
    fx_train, fx_val, fy_train, fy_val = train_test_split(fx_train, fy_train, test_size=0.3, random_state=random_state)

    for i, file in enumerate(fx_train):
        ori = Image.open(file).convert("L") 
        ori = ori.resize((image_size_x, image_size_y)) 
        ori = square(ori, 255)
        ori = pil2opencv(ori)
        hough(ori)
        ori = opencv2pil(ori)

        data = np.asarray(ori) 
        x_train.append(data)

        if fy_train[i] < ON_line_lower:
            y_train.append(0)
        elif fy_train[i] > ON_line_upper:
            y_train.append(0)
        else:
            y_train.append(1)

        params = {'rotation_range': 180, 'width_shift_range': 0.1, 'height_shift_range': 0.1, 'vertical_flip': True, 'horizontal_flip': True, 'shear_range': 2}
        datagen = image.ImageDataGenerator(**params)
        x = data[np.newaxis]  
        x = x[:, :, :, np.newaxis]  
        gen = datagen.flow(x, batch_size=1) 
        
        for n in range(9):
            batches = next(gen)  
            aug = np.squeeze(batches)
            x_train.append(aug) 
            if fy_train[i] < ON_line_lower:
                y_train.append(0)
            elif fy_train[i] > ON_line_upper:
                y_train.append(0)
            else:
                y_train.append(1)

            if DEBUG:
                gen_img = batches[0].astype(np.uint8)
                plt.subplot(3, 3, n+1)
                plt.imshow(gen_img)
                plt.axis('off')
                plt.gray()
                plt.tight_layout()
        plt.show()


    for i, file in enumerate(fx_test): 
        ori = Image.open(file).convert("L") 
        ori = ori.resize((image_size_x, image_size_y)) 
        ori = square(ori, 255)
        ori = pil2opencv(ori)
        hough(ori)
        ori = opencv2pil(ori)
        data = np.asarray(ori) 
        x_test.append(data) 
        
        if fy_test[i] < ON_line_lower:
            y_test.append(0)
        elif fy_test[i] > ON_line_upper:
            y_test.append(0)
        else:
            y_test.append(1)

    for i, file in enumerate(fx_val): 
        ori = Image.open(file).convert("L") 
        ori = ori.resize((image_size_x, image_size_y)) 
        ori = square(ori, 255)
        ori = pil2opencv(ori)
        hough(ori)
        ori = opencv2pil(ori)
        data = np.asarray(ori) 
        x_val.append(data)
        
        if fy_val[i] < ON_line_lower:
            y_val.append(0)
        elif fy_val[i] > ON_line_upper:
            y_val.append(0)
        else:
            y_val.append(1)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)

x_train = x_train.astype('float32')
x_train = x_train/255
x_test = x_test.astype('float32')
x_test= x_test/255
x_val = x_val.astype('float32')
x_val= x_val/255

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
y_val = to_categorical(y_val, 2)

train_newarray = (x_train.shape[0], x_train.shape[1], x_train.shape[1], 1)
x_train = np.reshape(x_train, train_newarray)

test_newarray = (x_test.shape[0], x_test.shape[1], x_test.shape[1], 1)
x_test = np.reshape(x_test, test_newarray)

valid_newarray = (x_val.shape[0], x_val.shape[1], x_val.shape[1], 1)
x_val = np.reshape(x_val, valid_newarray)

print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("x_val", x_val.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)
print("y_val", y_val.shape)

opt = optimizers.Adam(lr= 0.05) 

def model_train(x_train, y_train, x_val, y_val):
    
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),padding="same", input_shape=(x_train.shape[1], x_train.shape[2], 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3,3)))  

    model.add(Conv2D(64,(3,3),padding="same")) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),padding="same")) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3,3)))  

    model.add(Conv2D(256,(3,3),padding="same")) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(512,(3,3),padding="same")) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3,3)))  

    model.add(Flatten())

    model.add(Dense(512))                  
    model.add(Activation('relu'))  
    model.add(BatchNormalization())

    model.add(Dense(16))                                       
    model.add(Activation('relu'))    
    model.add(BatchNormalization())

    model.add(Dense(2)) 
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"]
                 )
    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("x_val", x_val.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)
    print("y_val", y_val.shape)

    early_stopping = [EarlyStopping(patience=20, verbose=1)]

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.9, patience=10, min_lr=0.0001)

    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), batch_size= 32, epochs= 200, shuffle=True, callbacks=[reduce_lr]) #, early_stopping

    model.save_weights('result/cnn_weights.h5') 
    model.save('result/cnn_model_weight.h5') 

    fig1 = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    fig1.savefig("Accuracy.jpg")

    fig2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show
    fig2.savefig("Loss.jpg")
    
    fig3 = plt.figure()
    plt.title("lr")
    plt.plot(range(len(history.history["lr"])), history.history["lr"])
    plt.show()
    fig3.savefig("lr.jpg")
  

    return model

def acc(y_label, y_pred):

    return np.mean(np.abs(y_label - y_pred) < 0.1)

def evaluate(model,x_test,y_test):
    scores = model.evaluate(x_test,y_test,verbose=1)
    print("Test Loss: ", scores[0])
    print("Test accuracy: ", scores[1])
                  

p = np.random.permutation(len(x_train))    
x_train, y_train = x_train[p], y_train[p]  
u = np.random.permutation(len(x_val))  
x_val, y_val = x_val[u], y_val[u]  
v = np.random.permutation(len(x_test))   
x_test, y_test = x_test[v], y_test[v] 

model = model_train(x_train, y_train, x_val, y_val)

evaluate(model ,x_test, y_test)

y_pred_train = model.predict(x_train)
y_pred_val = model.predict(x_val)
y_pred_test = model.predict(x_test)
print("pred", y_pred_test)
print("val", y_val)
print("test", y_test)

print("train")
predicted = np.argmax(y_pred_train, 1)
test = np.argmax(y_train, 1)
print(classification_report(test, predicted, target_names=["Retry", "Pass"]))
print("val")
predicted = np.argmax(y_pred_val, 1)
test = np.argmax(y_val, 1)
print(classification_report(test, predicted, target_names=["Retry", "Pass"]))
print("test")
predicted = np.argmax(y_pred_test, 1)
test = np.argmax(y_test, 1)
print(classification_report(test, predicted, target_names=["Retry", "Pass"]))

plt.scatter(y_train, y_pred_train, s=20, c="blue")
plt.scatter(y_val, y_pred_val, s=20, c="green")
plt.scatter(y_test, y_pred_test, s=20, c="red")
plt.legend(['Train', 'Validation', 'Test'])
plt.ylabel('Predict')
plt.xlabel('Test')
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.grid()
plt.show()






