# Módulo 8 — Sesión 3: Sentiment con Transformers

**Tarea.** Usar pipeline de Hugging Face para sentimiento en reseñas simuladas.  
Modelo sugerido: `nlptown/bert-base-multilingual-uncased-sentiment`.

## Instalación (solo esta carpeta)
```bash
pip install -r requirements.txt
```

## Pasos
1) Crear ≥10 reseñas de ejemplo (positivo/negativo/mixto).  
2) Usar `pipeline("sentiment-analysis")` y registrar **texto + predicción + score**.  
3) Guardar figuras/tablas en `reports/figures/` y enlazarlas aquí.

## Reflexión
Comparar con TF-IDF; ventajas y límites del enfoque con Transformers.

<!-- Badge de Colab (actívalo cuando subas el .ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/covalenzuela/bootcamp-ml-2025/blob/main/modules/m08_nlp/actividad_s03/notebooks/01_sentiment_transformers.ipynb
)
-->
