import pandas as pd

# Devuelve el archivo de salida con las predicciones
def formatPredictions(predictions_final):
  pd_predictions_final = pd.DataFrame(columns=["id", "tiene_alto_valor_adquisitivo"])
  pd_predictions_final.tiene_alto_valor_adquisitivo = predictions_final
  pd_predictions_final['id'] = pd_predictions_final.index + 1
  pd_predictions_final.set_index('id', inplace=True)
  return pd_predictions_final
  
def exportPredictions(df_predictions, file_name):
  df_predictions.to_csv(file_name + ".csv")