import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("rain.csv")
# print(data.head())
# print(data.info)
# print(data.describe)


data["datetime"]=pd.to_datetime(data["datetime"])

# print(data.isnull().sum)

data.ffill()


data['preciptype'] = data['preciptype'].map({'rain': 1, np.nan: 0})

data["year"]=data["datetime"].dt.year
data["month"]=data["datetime"].dt.month
data["day"]=data["datetime"].dt.day

# print(data.info)

# print(data.describe)


features = ['tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'precipprob', 
            'precipcover', 'windgust', 'windspeed', 'winddir',
            'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
            'solarenergy', 'uvindex', 'severerisk', 'month', 'day']

target=["preciptype"]



from sklearn.model_selection import train_test_split

x=data[features]
y=data[target].values.ravel()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=24)



from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train,y_train)
prediction=model.predict(x_test)

from sklearn import metrics
print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test,prediction))}")


#get features importance

importances=model.feature_importances_
indices=np.argsort(importances)[::-1]

#plot features importances

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(len(importances)),importances[indices],align="center")
plt.xticks(range(len(importances)),[features[i] for i in indices],rotation=90)
plt.tight_layout()
# plt.show()




from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5,10]
}

grid_search=GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid,cv=5,scoring="neg_mean_squared_error")
grid_search.fit(x_train,y_train)

print("DEst Parameters:",grid_search.best_params_)


def predict_weather(input_feature):
    input_df=pd.DataFrame([input_feature])

    for feature in features:
        if feature not in input_df.columns:
            input_df[feature]=data[feature].median()
    

    prediction=model.predict(input_df[features])
    return prediction [0]


example_input = {
    'tempmax': 75,
    'tempmin': 55,
    'temp': 65,
    'dew': 50,
    'humidity': 70,
    'precipprob': 50,
    'precipcover': 2,
    'preciptype': 1,
    'windgust': 15,
    'windspeed': 8,
    'winddir': 180,
    'sealevelpressure': 1015,
    'cloudcover': 50,
    'visibility': 10,
    'solarradiation': 200,
    'solarenergy': 15,
    'uvindex': 7,
    'severerisk': 30,
    'month': 6,
    'day': 15
}

predicted_precip = predict_weather(example_input)
print(f"Predicted precipitation: {'Rain' if predicted_precip == 1 else 'No rain'}")

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(len(importances)),importances[indices],align="center")
plt.xticks(range(len(importances)),[features[i] for i in indices],rotation=90)
plt.tight_layout()
# plt.show()