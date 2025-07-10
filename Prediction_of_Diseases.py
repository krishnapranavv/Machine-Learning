import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from statistics import mode, StatisticsError
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/krishnapranav/Git/Machine-Learning/improved_disease_dataset.csv')
le = LabelEncoder()
df["disease"] = le.fit_transform(df["disease"])

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#OVERSAMPLING
ros = RandomOverSampler(random_state=7)
X_resampled, y_resampled = ros.fit_resample(X, y)

#FILLING NIL WITH 0
X_resampled = X_resampled.fillna(0)

#SHAPING DF TO 1D (IF IT WAS NOT ALREADY)
if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

#Decision Tree & Random Forest
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

cv_scoring = 'accuracy'  
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    try:
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=stratified_kfold,
            scoring=cv_scoring,
            n_jobs=-1,
            error_score='raise' 
        )
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Accuracy: {scores.mean() * 100:.2f}%")
    except Exception as e:
        print(f"Model: {model_name} failed with error:")
        print(e)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_resampled,y_resampled)
rf_model = RandomForestClassifier()
rf_model.fit(X_resampled,y_resampled)

dt_pred = cross_val_predict(dt_model, X_resampled, y_resampled, cv=stratified_kfold)
rf_pred = cross_val_predict(rf_model, X_resampled, y_resampled, cv=stratified_kfold)

#SVM model
svm_model= SVC()
svm_model.fit(X_resampled,y_resampled)
svm_pred = svm_model.predict(X_resampled)

#cf_matrix = confusion_matrix(y_test, svm_pred)
#sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")
#plt.title("Confusion Matrix for SVM Classifier (on Test Data)")
#plt.ylabel("Actual")
#plt.show()

svm_accuracy = accuracy_score(y_resampled, svm_pred)
print(f"SVM Accuracy on Test Set: {svm_accuracy * 100:.2f}%")

#NAIVE BAYES
nb_model = GaussianNB()
nb_model.fit(X_resampled,y_resampled)
nb_pred= nb_model.predict(X_resampled)

nb_accuracy= accuracy_score(y_resampled,nb_pred)
print(f"Naive Bayes Accuracy on Test Set: {nb_accuracy * 100:.2f}%")

final_pred= [mode([i,j,k,l]) for i,j,k,l in  zip(dt_pred,rf_pred,svm_pred,nb_pred)]
symptoms = X.columns.values
symptom_index= {symptoms: idx for idx,symptoms in enumerate(symptoms)}

#DISEASE PREDICTION
def disease_prediction(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)
    
    for symptom in input_symptoms:
        if symptom in symptom_index :
            input_data[symptom_index[symptom]] = 1
    
    input_data = pd.DataFrame([input_data], columns=X.columns)

    dt_pred = le.classes_[dt_model.predict(input_data)[0]]
    rf_pred = le.classes_[rf_model.predict(input_data)[0]]
    svm_pred = le.classes_[svm_model.predict(input_data)[0]]
    nb_pred = le.classes_[nb_model.predict(input_data)[0]]

    try:
        final_pred = mode([dt_pred,rf_pred,svm_pred,nb_pred])
    except StatisticsError:
        final_pred = svm_pred
    
    return {
        "Decision Tree": dt_pred,
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

predictions = disease_prediction("Itching,Skin Rash,Nodal Skin Eruptions")
for model, prediction in predictions.items():
    print(f"{model}: {prediction}")