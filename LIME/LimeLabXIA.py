#https://arxiv.org/pdf/1602.04938v3.pdf

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Ejemplo de modelo complejo
from sklearn.datasets import make_classification  # Generar datos tabulares de ejemplo

# Función para generar datos perturbados alrededor de una instancia
def perturb_data(instance, num_samples=500):
    perturbed_data = np.random.randn(num_samples, instance.shape[0]) * 0.05  # Ruido pequeño
    perturbed_data += instance  # Añadir instancia original
    return perturbed_data

# Función para ponderar las muestras según su cercanía a la instancia original
def compute_weights(distances, kernel_width=0.25):
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

# Ejemplo de uso con datos tabulares
# Generar datos de ejemplo y entrenar un modelo complejo
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Seleccionar una instancia para explicar
instance = X_test[0]

# Generar datos perturbados y obtener predicciones del modelo para estos
perturbed_data = perturb_data(instance)
predictions = model.predict_proba(perturbed_data)[:, 1]  # Asumiendo clasificación binaria

# Calcular distancias y pesos
distances = euclidean_distances(perturbed_data, instance.reshape(1, -1)).reshape(-1)
weights = compute_weights(distances)

# Entrenar un modelo lineal para explicar
explainer_model = LinearRegression()
explainer_model.fit(perturbed_data, predictions, sample_weight=weights)

# Coeficientes del modelo lineal como explicación
print("Coeficientes de la explicación:", explainer_model.coef_)
