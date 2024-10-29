from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
import matplotlib.pyplot as plt


train_path='C:/Post Graduate Course in Data Analytics/SIGN LANGUAGE GESTURE RECOGNITION/train'
valid_path='C:/Post Graduate Course in Data Analytics/SIGN LANGUAGE GESTURE RECOGNITION/validation'

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
valid_datagen=ImageDataGenerator(rescale=1./255)

train_set=train_datagen.flow_from_directory(train_path,target_size=(64,64),batch_size=10,color_mode='grayscale',class_mode='categorical')

valid_set=train_set=valid_datagen.flow_from_directory(valid_path,target_size=(64,64),batch_size=10,color_mode='grayscale',class_mode='categorical')

model=Sequential()
model.add(Conv2D(8,kernel_size=(3,3),activation='relu',input_shape=(64,64,1)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(64,64,1)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(64,64,1)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics='acc')

history=model.fit_generator(train_set,validation_data=valid_set,epochs=7)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.legend(loc='best')
plt.show()

plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.legend(loc='best')
plt.show()

model.save('sign_language_gesture_recognition.h5')
model.save_weights('sign_language_categorical_weights.h5')