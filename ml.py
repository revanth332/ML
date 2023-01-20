from PIL import Image,ImageChops
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

def pixalize_img(img_path):
    image = Image.open(img_path,'r')
    image = image.convert('L')
    image = ImageChops.invert(image)
    image = image.resize((28,28))
    px_data = list(image.getdata())
    for i in range(len(px_data)):
        if px_data[i]/255 <= 0.43:
            px_data[i] = 0
    px_data = np.array(px_data)/255
    # image.show(px_data)
    return px_data

def Naive_Bayes_classifier(images):
    mnb = MultinomialNB()
    mnb.fit(x_train,y_train)
    # mnb.score(x_train,y_train)
    y_pred = mnb.predict(images)
    accuracy = mnb.score(x_test,y_test) 
    print("Naive_Bayes_classification: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",y_pred,end="\n")

def RandomForest_Classifier(images):
    rd = RandomForestClassifier()
    rd.fit(x_train,y_train)
    # rd.score(x_train,y_train)
    accuracy = rd.score(x_test,y_test) 
    rd_pred = rd.predict(images)
    print("RandomForest_Classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",rd_pred,end="\n")

def KNeighbors_Classifier(images):
    classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test,y_test)
    ans = classifier.predict(images)

    print("KNeighbors_Classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")

def DecisionTree_Classifier(images,type):
    ans = None
    if type=="gini":
        clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
        clf_gini.fit(x_train, y_train)
        accuracy = clf_gini.score(x_test,y_test)
        ans = clf_gini.predict(images)
    elif type=="entropy":
        clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
        clf_entropy.fit(x_train, y_train)
        accuracy = clf_entropy.score(x_test,y_test)
        ans = clf_entropy.predict(images)
    else:
        print("valid types are: 1)gini 2)entropy")
    print("DecisionTree_Classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")


def SVC_classifier(images):
    clf = SVC(kernel='linear') 
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test,y_test)
    ans = clf.predict(images)
    print("SVC_classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")


def Bagging_classifier(images):
    bagging = BaggingClassifier(KNeighborsClassifier(),
                                max_samples=0.5, max_features=0.5)
    bagging.fit(x_train,y_train)
    accuracy = bagging.score(x_test,y_test)
    ans = bagging.predict(images)
    print("Bagging_classifier: ")
    print("Accuracy :",round(accuracy,2))
    print("label for given image :",ans,end="\n")



images = ['number0.png','number1.png','number2.png','number3.png','number4.jpg','numbera.jpg']
pix_imgs = [pixalize_img(img_path) for img_path in images]

data= pd.read_csv(r"C:\Users\y20cb37\Desktop\digit-recognition\train.csv")
# test_data= pd.read_csv(r"C:\Users\USER\Desktop\Python\ML\data_test.csv")

x = data.iloc[:,1:].values
y = data.iloc[:,:1]["label"]

x=x/255

x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20,stratify=y)

# Naive_Bayes_classifier(pix_imgs)
# RandomForest_Classifier(pix_imgs)
# KNeighbors_Classifier(pix_imgs)
# DecisionTree_Classifier(pix_imgs,"gini")
# SVC_classifier(pix_imgs)
# Bagging_classifier(pix_imgs)4

Naive_Bayes_classifier([pix_imgs[4]])
RandomForest_Classifier([pix_imgs[4]])
KNeighbors_Classifier([pix_imgs[4]])
DecisionTree_Classifier([pix_imgs[4]],"entropy")
# SVC_classifier([pix_imgs[5]])
Bagging_classifier([pix_imgs[4]])

# # plt.imshow([pix_imgs[5]])
# print(pix_imgs[4])
plt.imshow(pix_imgs[4].reshape(28,28))
plt.show()
