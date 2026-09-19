import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

RANDOM_STATE = 42

# Simulamos 150 piezas de motos con 2 características (ej. peso y nivel de reflejo)
np.random.seed(RANDOM_STATE)
X = np.random.rand(150, 2) * 10
# Si la suma de características supera 12, requiere revisión/reposición (1), sino está OK (0)
y = (X[:, 0] + X[:, 1] > 12).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Muestras entrenamiento: {len(X_train)}")
print(f"Muestras prueba: {len(X_test)}")
print(f"Accuracy de clasificación de estado de piezas: {accuracy_score(y_test, pred):.3f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, pred))