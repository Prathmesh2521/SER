#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import numpy as np # linear algebra
import pandas as pd 
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , Dropout
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


paths =[]
labels=[]
for dirname, _, filenames in os.walk("D:\PRATHMESH\BE5\BE_Project\Doc\TESS Toronto emotional speech set data"):
    for filename in filenames:
        paths.append(os.path.join(dirname , filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
print('Dataset is Loaded')


# In[3]:


paths[700:705]


# In[4]:


labels[700:705]


# In[5]:


## Create a dataframe
df= pd.DataFrame()
df['speech']= paths
df['label'] =labels
df.head()


# In[6]:


df.shape


# In[7]:


df['label'].value_counts()


# In[8]:


sns.countplot(x = df['label'], data=df)


# In[9]:


def waveplot(data, sr , emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    
def spectrogram(data, sr ,emotion):
    x= librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time',y_axis='hz')
    plt.colorbar()


# In[10]:


emotion = 'fear'
path =np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[11]:


emotion = 'angry'
path =np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[12]:


emotion = 'disgust'
path =np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[13]:


emotion = 'neutral'
path =np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[14]:


emotion = 'sad'
path =np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[15]:


emotion = 'ps'
path =np.array(df['speech'][df['label']==emotion])[2]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[16]:


emotion = 'happy'
path =np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate= librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data , sampling_rate,emotion)
Audio(path)


# In[17]:


def extract_mfcc(filename):
    y, sr =librosa.load(filename, duration=5, offset=0.5)
    mfcc=np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
  
    return mfcc


# In[18]:


extract_mfcc(df['speech'][0]).shape


# In[19]:


X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc


# In[20]:


X = [x for x in X_mfcc]
X= np.array(X)
X.shape


# In[21]:


enc = OneHotEncoder()
y = enc.fit_transform(np.array(df['label']).reshape(-1,1)).toarray()
print(y)
y.shape


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=120, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[23]:


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
print(x_test.shape,x_train.shape)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_test.shape,x_train.shape)


# In[24]:


model = Sequential([
    LSTM(123, return_sequences=False,  input_shape=(x_train.shape[1],1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics='accuracy')
model.summary()


# In[25]:


history = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=50, batch_size=64)


# In[26]:


epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[27]:


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[28]:


pred_test = model.predict(x_test)
y_pred = enc.inverse_transform(pred_test)

Y_Test = enc.inverse_transform(y_test)
print(Y_Test , y_pred)


# In[29]:


df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = Y_Test.flatten()
df


# In[30]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, model(x_test))

print(r2)
#The result 0.98 shows that the model fits the testing set as well


# In[31]:


def predict_emotion(audio_file_path):
    mfcc = extract_mfcc(audio_file_path)
    X = np.array(mfcc)
    
    print(X.shape)
    X=X.reshape(1,40)
    # Standardize the features
    scaler = StandardScaler()
    X = np.expand_dims(X, -1)

    # Reshape the features
   

    # Predict the emotion
    pred = model.predict(X)
   
    emotion = enc.inverse_transform(pred)[0][0]
    return emotion
   


# In[34]:


audio_file_path ="D:\PRATHMESH\BE5\BE_Project\Doc\TESS Toronto emotional speech set data\OAF_neutral\OAF_base_neutral.wav"
predicted_emotion = predict_emotion(audio_file_path)
print('Predicted Emotion:', predicted_emotion)


# In[41]:


audio_file_path ="D:\PRATHMESH\BE5\BE_Project\Doc\TESS Toronto emotional speech set data\OAF_Sad\OAF_bite_sad.wav"
predicted_emotion = predict_emotion(audio_file_path)
print('Predicted Emotion:', predicted_emotion)


# In[47]:


audio_file_path ="D:\PRATHMESH\BE5\BE_Project\Doc\TESS Toronto emotional speech set data\OAF_angry\OAF_yearn_angry.wav"
predicted_emotion = predict_emotion(audio_file_path)
print('Predicted Emotion:', predicted_emotion)

