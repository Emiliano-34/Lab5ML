# Función para calcular la matriz de confusión
def confusion_matrix_binary(y_true, y_pred):
    """
    Calcula la matriz de confusión para un conjunto de datos binario.

    Parámetros:
    y_true -- Lista con las clases verdaderas.
    y_pred -- Lista con las clases predichas por el modelo.

    Retorna:
    Matriz de confusión en el formato [[TP, FN], [FP, TN]].
    """
    TP = FN = FP = TN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 1 and pred == 0:
            FN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 0 and pred == 0:
            TN += 1
    return [[TP, FN], [FP, TN]]

# Función para calcular Accuracy
def accuracy(cm):
    TP, FN = cm[0]
    FP, TN = cm[1]
    return (TP + TN) / (TP + TN + FP + FN)

# Función para calcular Error
def error(cm):
    return 1 - accuracy(cm)

# Función para calcular Precision (Positive Predictive Value)
def precision(cm):
    TP, FN = cm[0]
    FP, TN = cm[1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0

# Función para calcular Recall (True Positive Rate)
def recall(cm):
    TP, FN = cm[0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0

# Función para calcular True Negative Rate (TNR) o Especificidad
def true_negative_rate(cm):
    FP, TN = cm[1]
    return TN / (TN + FP) if (TN + FP) > 0 else 0

# Función para calcular False Positive Rate (FPR)
def false_positive_rate(cm):
    FP, TN = cm[1]
    return FP / (FP + TN) if (FP + TN) > 0 else 0

# Función para calcular False Negative Rate (FNR)
def false_negative_rate(cm):
    TP, FN = cm[0]
    return FN / (TP + FN) if (TP + FN) > 0 else 0

# Función para calcular F1-Score
def f1_score(cm):
    prec = precision(cm)
    rec = recall(cm)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

# Ejemplo de uso:
# Valores verdaderos (y_true) y predichos (y_pred)
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # Clases verdaderas
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # Clases predichas por el modelo

# Calculamos la matriz de confusión
cm = confusion_matrix_binary(y_true, y_pred)
print("Matriz de Confusión:", cm)

# Calcular y mostrar las métricas de desempeño
print("Accuracy:", accuracy(cm))
print("Error:", error(cm))
print("Precision (PPV):", precision(cm))
print("Recall (TPR):", recall(cm))
print("True Negative Rate (TNR):", true_negative_rate(cm))
print("False Positive Rate (FPR):", false_positive_rate(cm))
print("False Negative Rate (FNR):", false_negative_rate(cm))
print("F1-Score:", f1_score(cm))
