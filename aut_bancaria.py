import pandas as pd
import seaborn as sns
from tensorflow.contrib.layers import real_valued_column
from tensorflow.python.estimator import DNNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Leitura dos dados de entrada.
data = pd.read_csv("dados_autent_bancaria.txt")

# Padronização dos dados para preparar para a etapa de treinamento.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('class', axis=1))
final_data = pd.DataFrame(data=scaled_data, columns=data.columns[:-1])

# Criação dos conjuntos de treinamento e de testes.
x = final_data.to_numpy()
y = data['class'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Criação da rede neural.
classifier = DNNClassifier(hidden_units=[10, 20, 10],
                                 n_classes=2,
                                 feature_columns=[real_valued_column("", dimension=4)])

classifier.fit(x_train, y_train, steps=200, batch_size=20)
predictions = list(classifier.predict(X_test))

print('Confusion Matrix \n', confusion_matrix(y_test, predictions))
print('Classfication Report \n', classification_report(y_test, predictions))
print('Accuracy of our model -', accuracy_score(y_test, predictions))
