# Módulo 8 — Evaluación Modular: Clasificación de notas clínicas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/covalenzuela/bootcamp-ml-2025/blob/main/modules/m08_nlp/evaluacion_modular/notebooks/clasificacion_notas_clinicas.ipynb
)

**Objetivo.** Clasificar la **gravedad** de notas clínicas (leve / moderado / severo) con un baseline clásico (TF‑IDF + Naive Bayes) y una variante moderna con **BERT** (opcional).

## Datos
- **Archivo:** `dataset_clinico_simulado_200.csv` (simulado para uso educativo).
- **Columnas:** `texto_clinico`, `edad`, `genero`, `afeccion`, `gravedad`.
- **Clases:** leve, moderado, severo.
- **Split:** 80/20 estratificado (semilla 42) → 160 filas de train, 196 features TF‑IDF.
- **Stats del preprocesamiento:** longitud media ≈ **6.95** tokens; vocabulario ≈ **70.0**; palabras totales ≈ **1390.0**.

## Pipeline
1. Limpieza y normalización (regex, minúsculas, stopwords ES).
2. **TF‑IDF** sobre `texto_proc` y modelo **MultinomialNB**.
3. Métricas (accuracy, **macro‑F1**), matriz de confusión y **top‑k palabras** por clase (interpretabilidad).
4. (**Opcional**) **BERT**: `BertForSequenceClassification` + tokenizer; comparación NB vs BERT.
5. (**Opcional**) **Fairness** por subgrupos (p.ej. `genero`): accuracy/F1 macro por grupo.

## Resultados (completa al correr)
- **NB + TF‑IDF** → Accuracy: … · Macro‑F1: …  
- **BERT** (opcional) → Accuracy: … · Macro‑F1: …  
- Exporta **figuras** (matriz de confusión, barras por clase, comparativa NB vs BERT) a `reports/figures/` y enlázalas aquí.

## Cómo reproducir

### Instalación (solo esta carpeta)
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm  # si usarás lematización en español
```

### Ejecutar notebook
```bash
# desde la raíz del repo
jupyter lab  # o abre el .ipynb en Colab
# archivo:
modules/m08_nlp/evaluacion_modular/notebooks/clasificacion_notas_clinicas.ipynb
```

### Descarga del dataset (desde el notebook)
```python
import gdown
file_id = "1pWlNKW31-MDEr6D8eNICjkadwLGeX2BR"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "dataset_clinico_simulado_200.csv", quiet=False)
```

## Reproducibilidad
- Semillas fijas (`random_state=42`) y configuración determinista.
- Mantener versiones en `requirements.txt`. Si usas Torch CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Ética
Datos **simulados** y de uso docente; este proyecto **no es** un sistema de diagnóstico.
