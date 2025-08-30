\# AI-CUT 🎬

Asistente local de edición de vídeo con \*\*PyTorch + XGBoost\*\*.  

Detecta escenas, extrae features visuales y propone automáticamente segmentos “buenos” en formato compatible con \[LosslessCut](https://github.com/mifi/lossless-cut).



---



\## 📂 Estructura de carpetas



El sistema trabaja en:



C:\\ai-entrenamiento

├── dataset\\ # dataset.parquet acumulado para entrenamiento

├── labels\\ # copias archivadas de elegidos.csv con timestamp

├── models\\ # modelos entrenados (modelo\_xgb\_v1.json, v2…)

├── metrics.json # histórico de métricas (AUPRC, escenas entrenadas, timestamp)

└── \*.mp4, \*.csv # vídeos brutos y sus elegidos.csv



Cada carpeta donde hay un vídeo genera un directorio oculto `.aicache\\` con:

video.samples.parquet

video.scenes.parquet

video.meta.json



Y junto a cada vídeo bruto aparecen:

\- `video.mp4` → archivo original  

\- `video.csv` → exportado manual desde LosslessCut (elegidos.csv)  

\- `video.propuestos.csv` → generado por AI-CUT (segmentos propuestos)



---



\## 🚀 Uso



\### 1. Analizar un vídeo

```bash

python main.py analyze "video.mp4" --forcé



2\. Proponer segmentos

python main.py propose "video.mp4"





➡️ genera video.propuestos.csv listo para importar en LosslessCut.



3\. Ingestar un par vídeo/CSV (entrenamiento)

python main.py ingest "video.mp4"



4\. Re-entrenar modelo

python main.py train



5\. Procesar en batch

python main.py ingest-batch C:\\ai-entrenamiento

python main.py propose-batch C:\\mis\_videos


### 🖱️ Scripts por lotes (Windows .bat)

Para mayor comodidad, puedes usar los `.bat` incluidos en la raíz del repo:

- `propose.bat` → genera propuestos.csv para todos los vídeos de la carpeta indicada.  
- `reanalizar - todos formatos.bat` → fuerza el análisis de todos los vídeos soportados en la carpeta de entrenamiento.  
- `train.bat` → reentrena el modelo con el dataset acumulado.  

Ejecuta cualquiera con doble clic desde el Explorador de Windows.



⚙️ Requisitos



Windows + GPU NVIDIA (CUDA)



Python 3.10 o superior



Notas



El script guarda cachés intermedios en .aicache.



Si quieres re-analizar un vídeo ignorando cachés, usa --force.



ultralogger es opcional: si no está instalado, el script usa un logger simple.



La salida para LosslessCut siempre es un único propuestos.csv por vídeo.





