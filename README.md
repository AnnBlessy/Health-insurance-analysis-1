# Classification of Insurance Schemes based on the health status of the individuals 

## Importing all the necessary python libraries
```
import pandas as pd
import matplotlib.pyplot as plt
```

## Reading the Dataset
```
data = pd.read_csv("health_insurance_dataset.csv")
data.head()
```
![head](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/63f88728-05bc-4377-97fd-e094022ec728)

```
data.tail()
```
![tail](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/a177e18c-6988-4143-beb2-f0719c44b0ba)

```
data.isnull().sum()
```
![info](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/65194af3-ba4e-49a0-9813-2126fba75de8)

```
data.info()
```
![null](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/d910a7d1-4487-490f-bef8-b0a9d32ec7c5)


## Conversion of object type variables into numerical format 

```
data1 = data.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["Insurance_Scheme"] = le.fit_transform(data1["Insurance_Scheme"])
data1
```
![data-1](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/f48b29ac-4b2e-4a28-b5c0-73bb07666319)
![data-2](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/e5903b78-ad65-483a-84c1-1cd05cc53087)
![data-3](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/907ea86c-4e70-4418-b356-05db77b72211)


## Determining X values

```
x = data1.iloc[:,:-1]
print("Data-status")
x
```
![x-1](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/135cd38c-e40f-4f40-a154-9348fc846ec4)
![x-2](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/93b0b96f-19d0-4f33-bf30-7121371b2e08)
![x-3](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/13fdd735-b59b-4f76-8d95-5301003483d0)


## Determining Y values

```
y = data1["Insurance_Scheme"]
print("data-status")
y
```
![y](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/fb765bc6-cfe3-4d7b-844b-65785e051689)

![y-2](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/19be6da3-2fc5-4fb8-bb23-daf72768823e)


## Training and Testing of X and Y values

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 2/3,random_state = 0)
```

## Using Logistic Regression to classify the insurance schemes 

```
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(" y_prediction array")
y_pred
```
![y_pred](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/48deda5e-9c41-48d8-86fe-ec400df24fa3)

## Calculating the Accuracy rate

```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy
```
![acc](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/97ebe473-64e3-4438-aea4-89136f0595ac)


## Confusion Array

```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion
```
![ca](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/a26e75d4-9ef7-4fb4-a8b1-a74a4dab1121)

## Classification Report

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)
```
![cr](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/c6b99852-e71d-4f6f-a803-a52e18c437e5)

## Verifying by feeding user inputs(basic)

```
print("Predictions : ")
print("\nSET-1")
lr1=lr.predict([[40,24.478327,86,132,0,1]])
if lr1 == 0:
    print("Basic Insurance Scheme")
elif lr1 == 1:
    print("Standard Insurance Scheme")
else:
    print("Premium Insurance Scheme")

print("\nSET-2")
lr2=lr.predict([[36,24.027460,133,184,0,1]])
if lr2 == 0:
    print("Basic Insurance Scheme")
elif lr2 == 1:
    print("Standard Insurance Scheme")
else:
    print("Premium Insurance Scheme")

print("\nSET-3")
lr3=lr.predict([[45,22.18131,97,132,1,0]])
if lr3 == 0:
    print("Basic Insurance Scheme")
elif lr3 == 1:
    print("Standard Insurance Scheme")
else:
    print("Premium Insurance Scheme")
```
![OP](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/c85994f3-781e-4951-96fb-66431190d5cb)



# Health-insurance-analysis

## Importing all the necessary python libraries
```
import pandas as pd
import matplotlib.pyplot as plt
```

## Reading the Dataset
```
data = pd.read_csv("insurance.csv")
data.head()
```
![head](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/e9be24b3-8c6d-49b2-9849-de918a4d6ba7)

```
data.tail()
```
![tail](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/de2c3efc-46af-4332-9247-21e1ffd125f3)


## Dataset info
```
data.info()
```
![info](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/e01d31a5-0ed9-46d8-8659-9443662e4b47)

## Checking of Nulls
```
data.isnull().sum()
```
![nullcheck](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/8aae47c5-b1f2-469b-a46f-0b1fd54e72a8)

```
df = data.copy()
df = df.drop(["children","region"],axis = 1)
df.head()
```
![new](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/d0a6af06-74b4-4f7c-b471-6045cb9853da)


## Conversion of object type variables into categorical variables or numerical format 

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df
```
![d1](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/4d39ab67-0401-436c-ae2e-9c9b28734287)
![d2](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/1ae3d65a-b9ec-42ca-8b7d-dc6d803b652b)
![d3](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/45da87ce-ee21-4fd4-ad81-fa90a6b6d28f)

## Determining the optimum number of clusters using Elbow Graph 

```
from sklearn.cluster import KMeans
wcss=[]   #Within-Cluster Sum of Squares.

for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.show()
```
![graph](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/3518044d-48b0-483c-8621-8ad92f418b74)

## Performing K-Means Clustering

```
km = KMeans(n_clusters=4)
km.fit(df.iloc[:,3:])

y_pred=km.predict(df.iloc[:,3:])
y_pred
```
![km](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/8bc7bcb4-e00c-4881-8f00-6a0799afdba4)

## Plotting graph for insurance segments

```
df["cluster"]=y_pred
df0 = df[df["cluster"]==0]
df1 = df[df["cluster"]==1]
df2 = df[df["cluster"]==2]
df3 = df[df["cluster"]==3]

plt.scatter(df0["bmi"],df0["charges"],c="red",label="cluster0")
plt.scatter(df1["bmi"],df1["charges"],c="yellow",label="cluster1")
plt.scatter(df2["bmi"],df2["charges"],c="blue",label="cluster2")
plt.scatter(df3["bmi"],df3["charges"],c="magenta",label="cluster3")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.legend()
plt.title("Insurance Segments")
```
![OP_graph](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/14f07282-88ce-4cf7-91fd-7359506e2544)
