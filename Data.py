import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
data={
    "study_hours":[1,2,3,4,5],
    "Test_score":[40,50,60,70,80]
}
df=pd.DataFrame(data)
standard_scaler=StandardScaler()
standard_scaled=standard_scaler.fit_transform(df)
print("Standard scaler output")
print(pd.DataFrame(standard_scaled,columns=["study_hours","Test_score"]))

minmax_scaler=MinMaxScaler()
minmax_scaled=minmax_scaler.fit_transform(df)
print("\n MinMax sclaed output")
print(pd.DataFrame(minmax_scaled,columns=["study_hours","Test_score"]))

x=df[["study_hours"]]
y=df[["Test_score"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training data")
print(x_train)
print("Test_Data")
print(x_test)

