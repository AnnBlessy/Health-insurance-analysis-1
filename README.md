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
![head](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/b6adf397-685a-43bc-98a2-c12a482dd2df)

```
data.tail()
```
![tail](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/99080ed1-0a70-4f40-ae3f-dd9fbdc39b0e)

## Dataset info
```
data.info()
```
![info](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/e01d31a5-0ed9-46d8-8659-9443662e4b47)

## Checking of Nulls
```
data.isnull().sum()
```
![nullcheck](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/812f33af-acf5-4067-b87d-701ffc201d0d)

```
df = data.copy()
df = df.drop(["children","region"],axis = 1)
df.head()
```
![new](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/9392242f-56a2-4334-849d-bc31fd660f20)

## Conversion of object type variables into categorical variables or numerical format 

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df
```
![d1](https://github.com/AnnBlessy/Health-insurance-analysis-1/assets/119477835/3b34e2fd-3ef0-4e58-a233-1c8bf11fa5fa)
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
