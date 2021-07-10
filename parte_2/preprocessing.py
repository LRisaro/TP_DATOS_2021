import pandas as pd
from sklearn import preprocessing

#Funcion que devuelve un 1 si esa row posee esta combinacion en especifico de valores en sus columnas "trabajo" y "rol_familiar_registrado"
def set_value_row_casado_trabajo(row):
    if (row.rol_familiar_registrado == "casado" and (row.trabajo == "profesional_especializado" or row.trabajo == "directivo_gerente" )):
      return 1                 
    else:
      return 0

#Aplica one hot encoding
def apply_one_hot_encoding(df):  
    return pd.get_dummies(df, drop_first=True, columns=["categoria_de_trabajo", "educacion_alcanzada", "estado_marital", "religion", "rol_familiar_registrado", "trabajo"])
    
def feature_engineering_xg_rf(df):

    #Creo columna opera_en_bolsa que indica si opero en bolsa o no
    df["opera_en_bolsa"] = df["ganancia_perdida_declarada_bolsa_argentina"].apply(lambda x: 1 if x != 0 else 0)

    #Creo columna es_hombre que indica si la persona es hombre o no
    df["es_hombre"] = df["genero"].apply(lambda x: 1 if x == "hombre" else 0)

    #Unifico los labels "casado" y "casada" en "casado"
    df.rol_familiar_registrado.replace(to_replace=["casada"],  value=["casado"], inplace=True)

    #Creo columna casado_trabajo que indica si ese usuario posee una combinacion especifica entre la columna rol_familiar_registrado y trabajo
    df["casado_trabajo"] = df.apply(lambda row: set_value_row_casado_trabajo(row), axis= 1)

    #Dropeo columnas que no considero necesarias para la prediccion
    df.drop(columns = ["edad", "horas_trabajo_registradas", "barrio", "genero", "ganancia_perdida_declarada_bolsa_argentina", "anios_estudiados"], inplace=True)

    #Aplico one hot encoding a estas columnas en especifico  
    df = apply_one_hot_encoding(df)

    #Borro una columna de cada uno de los one hot creados
    df.drop(columns = ["categoria_de_trabajo_empleado_municipal", "educacion_alcanzada_1_anio", "estado_marital_matrimonio_civil", "religion_budismo", "rol_familiar_registrado_con_hijos", "trabajo_ejercito"], inplace=True)

    if "tiene_alto_valor_adquisitivo" in df.columns:
      label_encoder = preprocessing.LabelEncoder()
      label_encoder.fit(df.tiene_alto_valor_adquisitivo)

      X = df.drop(columns=['tiene_alto_valor_adquisitivo'])
      y = label_encoder.transform(df.tiene_alto_valor_adquisitivo)

      return X, y, df, label_encoder

    return df
    
    
#Devuelve solo las columnas que le pido a travez de los indexes
def get_columns_by_index(df, indexes):

  return df.iloc[:, lambda df: indexes]
  
#Convierto las probabilidades que devuele XGBoost para calcular las metricas
def get_int_predictions(preds):
  predictions = []

  for i in preds:
    if i < 0.6:
      predictions.append(0)
    else:
      predictions.append(1)

  return predictions
 