import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier  


# load the data
data = pd.read_csv('AlcoholOnly.csv')

# define features (X) and target (y)
#X = data.drop(columns=["MorphologyEye", "MorphologyNormal", "MorphologyTrunk", "MorphologyTail", "MorphologyLethal"])
X = data[["HeadMani", "TrunkMani", "TailMani", "PharynxMani"]]
#y = data[["MorphologyEye", "MorphologyNormal", "MorphologyTrunk", "MorphologyTail", "MorphologyLethal"]]
#y = data.drop(columns=["HeadMani", "TrunkMani", "TailMani", "PharynxMani"])
#y = data[["MorphologyLethal"]]
#y = data[["MorphologyPharynx"]]
#y = data[["MorphologyHead"]]
#y = data[["MorphologyTrunk"]]
y = data[["MorphologyTail"]]
#y = data[["MorphologyNormal"]]
#y = data[["MorphologyEye"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# base_classifier = DecisionTreeClassifier()
base_classifier = RandomForestClassifier(n_estimators=100)  

base_classifier.fit(X_train, y_train)

y_pred = base_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
accuracy_percentage = f"{accuracy * 100:.2f}%"

# report
report = classification_report(y_test, y_pred)
print(report)


# visualize confusion matrices for each label
mcm = multilabel_confusion_matrix(y_test, y_pred)
#for i, label in enumerate(["MorphologyEye", "MorphologyHead", "MorphologyTrunk", "MorphologyPharynx", "MorphologyLethal", "MorphologyNormal", "MorphologyTail"]):
#for i, label in enumerate(["MorphologyLethal"]):
#for i, label in enumerate(["MorphologyPharynx"]):
#for i, label in enumerate(["MorphologyHead"]):
#for i, label in enumerate(["MorphologyTrunk"]):
#for i, label in enumerate(["MorphologyTail"]):
#for i, label in enumerate(["MorphologyNormal"]):
for i, label in enumerate(["MorphologyTrunk"]):
    plt.title(f"RandomForest: Confusion Matrix for Alcohol {label} - accuracy: {accuracy_percentage}")

    sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

