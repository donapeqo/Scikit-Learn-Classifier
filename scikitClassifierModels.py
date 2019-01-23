from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

clf = tree.DecisionTreeClassifier()
clf2 = KNeighborsClassifier()
clf3 = RadiusNeighborsClassifier(radius=1000.0)
clf4 = SVC(kernel="linear", C=0.025,random_state=101)
clf5 = RandomForestClassifier(n_estimators= 10)
#ValueError: No neighbors found for test samples [0], you can try using larger radius,
# give a label for outliers, or consider removing them from your dataset.



# CHALLENGE - create 3 more classifiers...
# 1 DecisionTreeClassifier
# 2 KNeighborsClassifier
# 3 RadiusNeighborsClassifier
#4 SVC

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X,Y)
clf4 = clf4.fit(X,Y)
clf5 = clf5.fit(X,Y)


prediction = clf.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])
prediction4 = clf4.predict([[190, 70, 43]])
prediction5 = clf5.predict([[190, 70, 43]])


# CHALLENGE compare their reusults and print the best one!

print(prediction)
print(prediction2)
print(prediction3)
print(prediction4)
print(prediction5)
