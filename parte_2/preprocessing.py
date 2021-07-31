import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def GetSeedForTrainSplit():
    return 15;

#Crea una columna si el usuario pertenece a un cierto rango de edad
def set_age_range(df):
    df["edad_10_y_20"] = df["edad"].apply(lambda x: 1 if x > 10 and x <= 20 else 0)
    df["edad_21_y_30"] = df["edad"].apply(lambda x: 1 if x > 20 and x <= 30 else 0)
    df["edad_31_y_40"] = df["edad"].apply(lambda x: 1 if x > 30 and x <= 40 else 0)
    df["edad_41_y_50"] = df["edad"].apply(lambda x: 1 if x > 40 and x <= 50 else 0)
    df["edad_51_y_60"] = df["edad"].apply(lambda x: 1 if x > 50 and x <= 60 else 0)
    df["edad_61_y_70"] = df["edad"].apply(lambda x: 1 if x > 60 and x <= 70 else 0)
    df["edad_71_y_80"] = df["edad"].apply(lambda x: 1 if x > 70 and x <= 80 else 0)
    df["edad_81_y_90"] = df["edad"].apply(lambda x: 1 if x > 80 and x <= 90 else 0)
    
    return df

#Funcion que devuelve un 1 si esa row posee esta combinacion en especifico de valores en sus columnas "trabajo" y "rol_familiar_registrado"
def set_value_row_casado_trabajo(row):
    if (row.rol_familiar_registrado == "casado" and (row.trabajo == "profesional_especializado" or row.trabajo == "directivo_gerente" )):
      return 1                 
    else:
      return 0

#Aplica one hot encoding
def apply_one_hot_encoding(df, columnsToApply):  
  return pd.get_dummies(df, drop_first=True, columns = columnsToApply)
    
#Unifico los labels "casado" y "casada" en "casado"
def unificar_values_casado_casada(df):
    df.rol_familiar_registrado.replace(to_replace=["casada"],  value=["casado"], inplace=True)
    return df

#Creo columna es_hombre que indica si la persona es hombre o no
def crear_columna_es_hombre(df):
    df["es_hombre"] = df["genero"].apply(lambda x: 1 if x == "hombre" else 0)
    return df
    
def feature_engineering_xg_rf(df):

    #Creo columna opera_en_bolsa que indica si opero en bolsa o no
    df["opera_en_bolsa"] = df["ganancia_perdida_declarada_bolsa_argentina"].apply(lambda x: 1 if x != 0 else 0)

    df =  crear_columna_es_hombre(df)

    df = unificar_values_casado_casada(df)

    #Creo columna casado_trabajo que indica si ese usuario posee una combinacion especifica entre la columna rol_familiar_registrado y trabajo
    df["casado_trabajo"] = df.apply(lambda row: set_value_row_casado_trabajo(row), axis= 1)

    #Dropeo columnas que no considero necesarias para la prediccion
    df.drop(columns = ["edad", "horas_trabajo_registradas", "barrio", "genero", "ganancia_perdida_declarada_bolsa_argentina", "anios_estudiados"], inplace=True)

    #Aplico one hot encoding a estas columnas en especifico  
    df = apply_one_hot_encoding(df,["categoria_de_trabajo", "educacion_alcanzada", "estado_marital", "religion", "rol_familiar_registrado", "trabajo"])

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

def feature_engineering_KNN_con_seleccion(df):
  df_clean = df.copy()   

  # Conversion de variables
  df_clean.rol_familiar_registrado.replace(to_replace=["casada"], value=["casado"], inplace=True)

  # Elimino columnas
  df_clean.drop(columns = ['barrio', 'anios_estudiados', 'edad', 'educacion_alcanzada', 'estado_marital', 
    'ganancia_perdida_declarada_bolsa_argentina', 'horas_trabajo_registradas', 'religion', 'categoria_de_trabajo' ], inplace = True)

  # Tratamiento de NaN
  df_clean.fillna('NaN', inplace=True)
    
  # One hot encoding
  df_clean = apply_one_hot_encoding(df_clean, ['genero', 'rol_familiar_registrado', 'trabajo'])
  
  return df_clean

def preprocessing_KNN_con_todos_los_features(df):
  df_clean = df.copy()

  # Conversion de variables
  df_clean["educacion_alcanzada"].replace({"preescolar" : 1, "1-4_grado": 2, "5-6_grado": 3, "7-8_grado" : 4, "9_grado" : 5, "1_anio" : 6, "2_anio" : 7, "3_anio" : 8, "4_anio" : 9, "5_anio" : 10, "universidad_1_anio" : 11, 
    "universidad_2_anio" : 12, "universidad_3_anio" : 13, "universidad_4_anio" : 14, "universiada_5_anio" : 15, "universiada_6_anio" : 16}, inplace = True)

  df_clean.rol_familiar_registrado.replace(to_replace=["casada"], value=["casado"], inplace=True)
  df_clean["opera_en_bolsa"] = df_clean["ganancia_perdida_declarada_bolsa_argentina"].apply(lambda x: 1 if x != 0 else 0)

  # Elimino columnas
  df_clean.drop(columns = ['ganancia_perdida_declarada_bolsa_argentina', 'barrio'], inplace = True)

  # Tratamiento de NaN
  df_clean.fillna('NaN', inplace=True)

  # One hot encoding
  df_clean = apply_one_hot_encoding(df_clean, ['categoria_de_trabajo', 'estado_marital', 'genero', 'religion', 'rol_familiar_registrado', 'trabajo'])

  return df_clean
  