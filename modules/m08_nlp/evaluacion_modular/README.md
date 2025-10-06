# Módulo 8 — Evaluación Modular: Clasificación de notas clínicas

**Objetivo.** Clasificar gravedad (**leve/moderado/severo**) con NLP.  
**Enfoques.** TF-IDF + modelo lineal (baseline) y BERT (opcional).  
**Ética.** Datos simulados; no diagnóstico.

## Instalación (solo este módulo)
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm  # si usarás lematización en español
```

## Ejecución
1) Abrir `notebooks/01_clasificacion_notas_clinicas.ipynb` (Colab o local).  
2) Pipeline: **EDA → Preproc → TF-IDF → Modelo lineal → (BERT opc.) → Métricas → Figuras**.  
3) Exportar figuras a `reports/figures/` y enlazarlas aquí.

## Resultados (placeholder)
- Macro-F1 (TF-IDF + lineal): …  
- Macro-F1 (BERT base): …

## Próximos pasos
Balance de clases, limpieza de abreviaturas clínicas, fine-tuning ligero de BERT.

<!-- Badge de Colab (actívalo cuando subas el .ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/covalenzuela/bootcamp-ml-2025/blob/main/modules/m08_nlp/evaluacion_modular/notebooks/01_clasificacion_notas_clinicas.ipynb
)
-->
