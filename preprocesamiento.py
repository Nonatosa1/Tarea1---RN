"""
Preprocesamiento e ingeniería de características para Spaceship Titanic.

Se procesan train y test juntos para que la imputación por grupo / apellido
pueda aprovechar toda la información disponible (sin usar la variable objetivo).
"""
import numpy as np
import pandas as pd

GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _imputar_por_clave(df, col, clave):
    """Rellena `col` con el valor más frecuente dentro de cada grupo `clave`."""
    moda = (df.dropna(subset=[col, clave])
              .groupby(clave)[col]
              .agg(lambda s: s.mode().iloc[0]))
    faltan = df[col].isna() & df[clave].notna()
    df.loc[faltan, col] = df.loc[faltan, clave].map(moda)


def construir_caracteristicas(train: pd.DataFrame, test: pd.DataFrame):
    n_train = len(train)
    df = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)

    # --- Descomposición de campos compuestos -------------------------------
    df["Grupo"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["TamGrupo"] = df.groupby("Grupo")["Grupo"].transform("count")
    df["Solo"] = (df["TamGrupo"] == 1).astype(float)

    cab = df["Cabin"].str.split("/", expand=True)
    df["Cubierta"], df["NumCabina"], df["Lado"] = cab[0], cab[1].astype(float), cab[2]

    df["Apellido"] = df["Name"].str.split().str[-1]
    df["TamFamilia"] = df.groupby("Apellido")["Apellido"].transform("count")

    # --- Imputación lógica --------------------------------------------------
    # Los miembros de un mismo grupo comparten planeta, cubierta y lado.
    for col in ["HomePlanet", "Cubierta", "Lado", "Destination"]:
        _imputar_por_clave(df, col, "Grupo")
    # Las familias (apellido) también comparten planeta de origen.
    _imputar_por_clave(df, "HomePlanet", "Apellido")

    # Quien está en criosueño no puede gastar.
    for g in GASTOS:
        df.loc[df["CryoSleep"] == True, g] = df.loc[df["CryoSleep"] == True, g].fillna(0)  # noqa: E712
    gasto_parcial = df[GASTOS].sum(axis=1, skipna=True)
    # Quien gastó algo no estaba en criosueño; quien no gastó nada probablemente sí.
    falta_cryo = df["CryoSleep"].isna()
    df.loc[falta_cryo, "CryoSleep"] = gasto_parcial[falta_cryo] == 0
    df["CryoSleep"] = df["CryoSleep"].astype(float)

    # Gastos restantes: mediana (casi siempre 0).
    for g in GASTOS:
        df[g] = df[g].fillna(df[g].median())
    df["VIP"] = df["VIP"].map({True: 1.0, False: 0.0}).fillna(0.0)
    df["Age"] = df["Age"].fillna(df.groupby("HomePlanet")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Números de cabina faltantes: mediana de su grupo, si no, global.
    df["NumCabina"] = df["NumCabina"].fillna(df.groupby("Grupo")["NumCabina"].transform("median"))
    df["NumCabina"] = df["NumCabina"].fillna(df["NumCabina"].median())

    for col in ["HomePlanet", "Cubierta", "Lado", "Destination"]:
        df[col] = df[col].fillna("Desconocido")

    # --- Características derivadas -----------------------------------------
    df["GastoTotal"] = df[GASTOS].sum(axis=1)
    df["SinGasto"] = (df["GastoTotal"] == 0).astype(float)
    for g in GASTOS + ["GastoTotal"]:
        df[g] = np.log1p(df[g])          # los gastos están muy sesgados
    df["Menor"] = (df["Age"] < 13).astype(float)
    df["Adolescente"] = ((df["Age"] >= 13) & (df["Age"] < 18)).astype(float)
    # Región de la nave según el número de cabina (bloques de 300).
    df["RegionCabina"] = (df["NumCabina"] // 300).clip(upper=6).astype(int).astype(str)

    categoricas = ["HomePlanet", "Destination", "Cubierta", "Lado", "RegionCabina"]
    numericas = ["Age", "NumCabina", "TamGrupo", "TamFamilia", *GASTOS, "GastoTotal"]
    binarias = ["CryoSleep", "VIP", "Solo", "SinGasto", "Menor", "Adolescente"]

    X = pd.get_dummies(df[categoricas], dtype=float)
    X = pd.concat([df[numericas].astype(float), df[binarias], X], axis=1)
    X["TamFamilia"] = X["TamFamilia"].fillna(1)

    # Estandarización usando estadísticas de entrenamiento.
    media = X.iloc[:n_train][numericas].mean()
    desv = X.iloc[:n_train][numericas].std() + 1e-8
    X[numericas] = (X[numericas] - media) / desv

    X_train = X.iloc[:n_train].to_numpy(dtype=np.float64)
    X_test = X.iloc[n_train:].to_numpy(dtype=np.float64)
    y_train = train["Transported"].astype(float).to_numpy()
    return X_train, y_train, X_test, list(X.columns)
