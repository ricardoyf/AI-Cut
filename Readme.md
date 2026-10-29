# AI-Cut

Asistente local de edición de vídeo para Windows. Analiza vídeos, aprende de selecciones previas y propone segmentos para LosslessCut.

## Estado

Versión: **main actual** (sin versión numerada publicada).

## Funciones

- Analiza vídeos y guarda muestras y escenas en una caché local.
- Ingiere segmentos elegidos manualmente como entrenamiento.
- Entrena modelos XGBoost a partir del histórico.
- Exporta propuestas como `video.propuestos.csv`.
- Procesa vídeos y carpetas por lotes.
- Mantiene modelos, métricas y etiquetas en `C:\\ai-entrenamiento`.

## Ejecutar

```powershell
pip install -r requirements.txt
python main.py --help
python main.py analyze "video.mp4" --force
python main.py propose "video.mp4"
```

Script principal: [`main.py`](./main.py).

