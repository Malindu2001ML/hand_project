import numpy as np
import os
import cv2
datapath = "dataset"
categories = os.listdir(datapath)
labels = [i for i in range(len(categories))]
category_dict = {"a":0,"ae":1,"e":2,"u":3}

data =[]
target = []

for category in categories:
    imgs_path = os.path.join(datapath,category)
    img_names = os.listdir(imgs_path)

    for img_name in img_names:
        img_path = os.path.join(imgs_path,img_name)
        img = cv2.imread(img_path,0)
        img = cv2.resize(img,(8,8))
        data.append(img)
        target.append(category_dict[category])

data = np.array(data)
data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
target = np.array(target)
np.save("data",data)
np.save("target",target)

data = np.load('data.npy')
target = np.load('target.npy')

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data,target,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_data,train_target)

predict_target = model.predict(test_data)
from sklearn.metrics import accuracy_score
acc = accuracy_score(test_target,predict_target)
print(acc)