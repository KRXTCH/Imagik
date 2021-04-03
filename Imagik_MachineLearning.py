from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import cv2
import numpy as np
import glob 
import os.path

# Directory and file detection using path name
def list(path): 
    fichier=[] 
    l = glob.glob(path+'\\*')
    for i in l: 
        if os.path.isdir(i):fichier.extend(list(i))
        else: fichier.append(i)     
    return fichier  # Return a string array that contains the name of different path file

# Preparation of both matrix X(arx) and y(ary)
def dataPreparation(file):
    arx = []
    ary = []
    for i in range(0, len(file)):
        img = cv2.imread(file[i])                       # Read the image at the i'th position
        dsize = (100, 100)                              #
        img = cv2.resize(img, dsize)                    # Resize it
        arx.append(img)                                 # Add it to our array
                                                        # _______________creation of our y matrix ________________
        if 'ColorClassification\Black'  in file[i]:     # Just detect the path name and add a value in our y array
            ary.append(0)                               #
        if 'ColorClassification\Blue'   in file[i]:     #
            ary.append(1)                               #
        if 'ColorClassification\Brown'  in file[i]:     #
            ary.append(2)                               #
        if 'ColorClassification\Green'  in file[i]:     #
            ary.append(3)                               #
        if 'ColorClassification\Orange' in file[i]:     #
            ary.append(4)                               #
        if 'ColorClassification\Red'    in file[i]:     #
            ary.append(5)                               #
        if 'ColorClassification\Violet' in file[i]:     #
            ary.append(6)                               #
        if 'ColorClassification\White'  in file[i]:     #
            ary.append(7)                               #
        if 'ColorClassification\Yellow' in file[i]:     #______________________________________________________
            ary.append(8)                               # 
    arx = np.array(arx)                                 # Transorm my final X array to a numpy array ( matrix )
    arx = arx.reshape(107,-1)                           # Reshape it for cast to 2dim 
    arx = arx/255.0                                     # Divide all my matrix so that values are between 0 and 1
    ary = np.array(ary)                                 # Transform my fianel y array to a numpy array ( matrix )
                                                        #
    return arx, ary                                     # Return both matrix 

# Preparation of the model that we gonna use ( using matrix X and y )
def modelPreparation(file):                                                                  
    X, y = dataPreparation(file)                                                             
                                                                                             
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # Split matrix into random train and test subsets
                                                                                             #
    random_forest_classifier = RandomForestClassifier(n_estimators=20, max_depth=10)         # Creation of n Forest Classifier
    random_forest_classifier.fit(X_train,y_train)                                            # Build a forest of trees from the training set ( X, y )
                                                                                             #
    y_pred = random_forest_classifier.predict(X_test)                                        # Predict class for X
    print("Model Accuracy :",metrics.accuracy_score(y_test, y_pred))                         # Accuracy classification score
                                                                                             #
    return random_forest_classifier                                                          # Return my model

# Treatments for our new image
def newDataTreatment(img_name):     # ___ We apply the same processing of our X matrix on our new image and return it ___
    data = cv2.imread(img_name)     #
    dsize = (100, 100)              #
                                    #
    data = cv2.resize(data, dsize)  #
    data = np.array(data)           #
    data = data.reshape(1, -1)      #
    data = data/255.0               #
                                    #
    return data                     # Return ou matrix

#Testing new image on our model
def testNewData(random_forest_classifier):
    colorName = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Violet', 'White', 'Yellow']
    
    data_one = newDataTreatment("images_test\\1.jpg")                           # Using 1st image
    data_two = newDataTreatment("images_test\\5.jpg")                           # Using 5th image
                                                                                #
    res_data_one = random_forest_classifier.predict(data_one)                   # Apply our model
    res_data_two = random_forest_classifier.predict(data_two)                   # on our images
                                                                                #
    print("The color of the first image is %s" % colorName[res_data_one[0]])    # Print the result of our prediction
    print("The color of the second image is %s" % colorName[res_data_two[0]])



## ___Main___ ##
    
file = list('ColorClassification')
model = modelPreparation(file)
testNewData(model)


