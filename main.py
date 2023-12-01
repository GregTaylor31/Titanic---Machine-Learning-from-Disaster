import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import os

def get_data():
    for dirname, _, filenames in os.walk('/venv/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    train_data = pd.read_csv("venv/input/train.csv")
    train_data.head()

    test_data = pd.read_csv("venv/input/test.csv")
    test_data.head()

    return test_data, train_data

def get_percentage_male_female_survival(train_data):
    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    rate_women = sum(women) / len(women)
    print("% of women who survived:", rate_women)

    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    rate_men = sum(men) / len(men)
    print("% of men who survived:", rate_men)

def random_forest_model(test_data, train_data):
    y = train_data["Survived"]

    train_data["Family Size"] = train_data["SibSp"] + train_data["Parch"] + 1
    test_data["Family Size"] = test_data["SibSp"] + test_data["Parch"] + 1

    features = ["Pclass", "Sex", "SibSp", "Parch", "Family Size"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

    return output

test_data, _ = get_data()
_, train_data = get_data()

get_percentage_male_female_survival(train_data)
output = random_forest_model(test_data, train_data)

try:
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")
except Exception as e:
    print(f"Error: {e}")