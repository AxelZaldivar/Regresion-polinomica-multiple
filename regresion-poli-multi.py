import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Leer datos desde un archivo CSV
with open('Fish.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Leer la primera fila como encabezado
    data = [row for row in reader]
data = np.array(data)

# Obtener las características y la variable objetivo
x1 = data[:, 0]
features = data[:, 6].astype(float)
y = data[:, 1].astype(float)

# Label Encoding para la variable categórica "x1" que son las Species
unique_x1 = np.unique(x1)
x1_mapping = {x1: i for i, x1 in enumerate(unique_x1)}
encoded_x1 = np.array([x1_mapping[specie] for specie in x1])
features_encoded = np.column_stack((encoded_x1, features))

# Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(features_encoded[:, 1], y, test_size=0.2, random_state=42)

# Normalizar las características de entrenamiento y prueba
mean_train = np.mean(X_train.reshape(-1, 1), axis=0)
std_train = np.std(X_train.reshape(-1, 1), axis=0)
X_train_normalized = ((X_train.reshape(-1, 1)) - mean_train) / std_train
X_test_normalized = (X_test - mean_train) / std_train

# Realizar regresión polinómica múltiple con descenso de gradiente
degree = 4
learning_rate=0.01
epochs=10000
X = np.column_stack([np.ones(X_train_normalized.shape[0])] + [X_train_normalized ** i for i in range(1, degree + 1)])
coefficients = np.random.rand(degree + 1)

# Descenso de gradiente
for epoch in range(epochs):
    predictions = X @ coefficients
    errors = predictions - y_train
    gradients = 2 * X.T @ errors / len(y_train)
    coefficients -= learning_rate * gradients

# Imprimir el error cuadrático medio en el conjunto de prueba
X_test_poly = np.column_stack([np.ones(X_test_normalized.shape[0])] + [X_test_normalized ** i for i in range(1, degree + 1)])
predictions_test = X_test_poly @ coefficients
mse_test = np.mean((predictions_test - y_test) ** 2)

# Graficar resultados en el conjunto de prueba
x_values = np.linspace(min(X_test_normalized), max(X_test_normalized), 100)
y_values = np.polyval(coefficients[::-1], x_values)

# Graficar
plt.scatter(X_test_normalized, y_test, label='Data Points')
plt.plot(x_values, y_values, label=f'Regression Line (Degree {degree})', color='red')
plt.title('Regresión Polinomial Multipler\nError cuadratico medio: ' + str(mse_test))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()