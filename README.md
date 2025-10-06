# Bootcamp ML — Monorepo (2025)

Monorepo de actividades y evaluaciones del bootcamp (SENCE).  
Cada módulo vive en `modules/<modulo>/...`.

## Requisitos base (repo completo)
```bash
pip install -r requirements.txt
```

## Estructura
- `modules/m08_nlp/evaluacion_modular/` — Clasificación de notas clínicas (Módulo 8)
- `modules/m08_nlp/actividad_s03/`    — Sentiment con Transformers (Sesión 3)
- `shared/` — utilidades compartidas

## Cómo trabajar por módulo
1. Ir a la carpeta del módulo.
2. Instalar sus dependencias (hereda del requirements base).
3. Abrir el notebook y seguir el pipeline (**EDA → Preproc → Modelo → Métricas → Figuras**).

## Notas de privacidad
Datos simulados/anonimizados; **no** es diagnóstico.
