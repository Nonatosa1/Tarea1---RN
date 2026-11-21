"""
Spaceship Titanic - Clasificación con una red neuronal (MLP) hecha en NumPy.

Pasos:
  1. Verificación numérica del gradiente (comprueba la retropropagación).
  2. Entrenamiento con partición entrenamiento/validación -> curvas de aprendizaje.
  3. Comparación contra modelos de referencia de scikit-learn.
  4. Validación cruzada estratificada de 5 pliegues; el ensamble de las 5 redes
     genera el archivo submission.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocesamiento import construir_caracteristicas
from red_neuronal import RedNeuronal, entropia_cruzada

DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(DIR, "resultados")
os.makedirs(SALIDA, exist_ok=True)

HIPERPARAMETROS = dict(capas_ocultas=(128, 64, 32), tasa_aprendizaje=5e-4, l2=1e-3, dropout=0.3)
ENTRENAMIENTO = dict(epocas=300, tam_lote=64, paciencia=25)


def verificar_gradiente(n_entradas=6):
    """Compara el gradiente analítico con diferencias finitas en una red pequeña."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(20, n_entradas)), rng.integers(0, 2, 20).astype(float)
    red = RedNeuronal(n_entradas, (5, 4), l2=0.0, dropout=0.0, semilla=1)
    sal, act, zs, masc = red._propagar(X)
    gW, _ = red._retropropagar(y, sal, act, zs, masc)
    max_err, h = 0.0, 1e-5
    for i, W in enumerate(red.W):
        for idx in np.ndindex(W.shape):
            orig = W[idx]
            W[idx] = orig + h; lp = entropia_cruzada(y, red.predecir_proba(X))
            W[idx] = orig - h; lm = entropia_cruzada(y, red.predecir_proba(X))
            W[idx] = orig
            num = (lp - lm) / (2 * h)
            max_err = max(max_err, abs(num - gW[i][idx]) / max(1e-8, abs(num) + abs(gW[i][idx])))
    return max_err


def graficar_curvas(hist, ruta):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(hist["perdida_train"]) + 1)
    ax[0].plot(ep, hist["perdida_train"], label="Entrenamiento")
    ax[0].plot(ep, hist["perdida_val"], label="Validación")
    ax[0].set(title="Pérdida (entropía cruzada)", xlabel="Época", ylabel="Pérdida")
    ax[1].plot(ep, hist["acc_train"], label="Entrenamiento")
    ax[1].plot(ep, hist["acc_val"], label="Validación")
    ax[1].set(title="Exactitud", xlabel="Época", ylabel="Exactitud")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(ruta, dpi=120); plt.close(fig)


def graficar_confusion(cm, ruta):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["No transp.", "Transp."]); ax.set_yticks([0, 1], ["No transp.", "Transp."])
    ax.set(xlabel="Predicción", ylabel="Real", title="Matriz de confusión (validación)")
    fig.tight_layout(); fig.savefig(ruta, dpi=120); plt.close(fig)


def main():
    train = pd.read_csv(os.path.join(DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DIR, "test.csv"))
    X, y, X_test, columnas = construir_caracteristicas(train, test)
    print(f"Datos: {X.shape[0]} entrenamiento, {X_test.shape[0]} prueba, {X.shape[1]} características")

    # 1) Verificación de gradiente --------------------------------------------
    err = verificar_gradiente()
    print(f"\n[1] Verificación de gradiente: error relativo máximo = {err:.2e} "
          f"({'OK' if err < 1e-5 else 'REVISAR'})")

    # 2) Partición entrenamiento / validación ----------------------------------
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    red = RedNeuronal(X.shape[1], semilla=42, **HIPERPARAMETROS)
    red.entrenar(X_tr, y_tr, X_va, y_va, **ENTRENAMIENTO)
    pred_va = red.predecir(X_va)
    acc_red = accuracy_score(y_va, pred_va)
    print(f"\n[2] Red neuronal (NumPy) - exactitud en validación: {acc_red:.4f} "
          f"(paró en la época {red.epoca_final})")
    print(classification_report(y_va, pred_va, target_names=["No transportado", "Transportado"], digits=4))
    graficar_curvas(red.historial, os.path.join(SALIDA, "curvas_aprendizaje.png"))
    graficar_confusion(confusion_matrix(y_va, pred_va), os.path.join(SALIDA, "matriz_confusion.png"))

    # 3) Modelos de referencia --------------------------------------------------
    referencias = {
        "Regresión logística": LogisticRegression(max_iter=2000, C=1.0),
        "MLP scikit-learn": MLPClassifier((128, 64, 32), alpha=1e-3, early_stopping=True,
                                          max_iter=500, random_state=42),
    }
    tabla = [("Red neuronal NumPy (propia)", acc_red)]
    for nombre, modelo in referencias.items():
        modelo.fit(X_tr, y_tr)
        tabla.append((nombre, accuracy_score(y_va, modelo.predict(X_va))))
    print("[3] Comparación en el mismo conjunto de validación:")
    for nombre, acc in tabla:
        print(f"    {nombre:<30s} {acc:.4f}")

    # 4) Validación cruzada + ensamble -----------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs, proba_test = [], np.zeros(len(X_test))
    for k, (i_tr, i_va) in enumerate(skf.split(X, y), 1):
        m = RedNeuronal(X.shape[1], semilla=k, **HIPERPARAMETROS)
        m.entrenar(X[i_tr], y[i_tr], X[i_va], y[i_va], **ENTRENAMIENTO)
        accs.append(accuracy_score(y[i_va], m.predecir(X[i_va])))
        proba_test += m.predecir_proba(X_test) / skf.n_splits
        print(f"    pliegue {k}: exactitud = {accs[-1]:.4f}")
    print(f"[4] Validación cruzada 5 pliegues: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": proba_test > 0.5})
    sub.to_csv(os.path.join(DIR, "submission.csv"), index=False)
    print(f"\nArchivo submission.csv generado ({len(sub)} filas, "
          f"{sub['Transported'].mean():.1%} predichos como transportados).")

    with open(os.path.join(SALIDA, "metricas.txt"), "w", encoding="utf-8") as f:
        f.write(f"Error relativo de verificacion de gradiente: {err:.2e}\n")
        f.write(f"Hiperparametros: {HIPERPARAMETROS} {ENTRENAMIENTO}\n\n")
        f.write("Exactitud en validacion (20% estratificado):\n")
        for nombre, acc in tabla:
            f.write(f"  {nombre:<30s} {acc:.4f}\n")
        f.write(f"\nValidacion cruzada 5 pliegues (red propia): {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n")
        f.write("Por pliegue: " + ", ".join(f"{a:.4f}" for a in accs) + "\n")


if __name__ == "__main__":
    main()
