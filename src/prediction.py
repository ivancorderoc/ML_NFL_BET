import catboost
import pandas as pd

# Cargar el modelo entrenado (archivo previamente guardado)
def cargar_modelo():
    return catboost.CatBoostClassifier().load_model('modelo_catboost.cbm')

# Función para realizar predicciones
def hacer_prediccion(modelo, datos_nuevos):
    return modelo.predict_proba(datos_nuevos)[:, 1]

if __name__ == "__main__":
    # Cargar el modelo entrenado
    modelo_catboost = cargar_modelo()

    # Ejemplo de datos nuevos para hacer una predicción
    datos_nuevos = pd.DataFrame({
        # Aquí ingresa los datos necesarios para realizar la predicción
        # Ejemplo: 'feature_1': valor_1, 'feature_2': valor_2, ...
    })

    # Realizar la predicción
    probabilidad_prediccion = hacer_prediccion(modelo_catboost, datos_nuevos)

    # Imprimir la probabilidad de predicción
    print(f"Probabilidad de resultado positivo: {probabilidad_prediccion}")
