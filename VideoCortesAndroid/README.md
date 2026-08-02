# VideoCortes

APK Android para recortar videos con flujo rapido tipo LosslessCut:
P reproducir, I inicio, O final, E exportar y N siguiente.

Version 1.6:
- Barra de desplazamiento manual para revisar libremente el video antes de exportar.
- La barra visual de segmentos tambien permite tocar/arrastrar para saltar a cualquier punto.

Version 1.5:
- Reproductor integrado con PlayerView oficial de Media3, mas parecido a Just Player internamente.
- Evita enganchar ExoPlayer a mano a TextureView.

Version 1.4:
- Reproductor real con AndroidX Media3/ExoPlayer, video fluido y audio.
- Mantiene la barra de segmentos y la exportacion rapida que ya funcionaba.

Version 1.3:
- Visor propio por fotogramas con MediaMetadataRetriever para evitar pantalla negra cuando el reproductor nativo no pinta el video.
- El boton P avanza la vista previa visual; I/O/E siguen usando esos tiempos para exportar.

Version 1.2:
- Reproduccion con SurfaceView + MediaPlayer para arrancar de forma mas estable.
- Version visible en una esquina de la aplicacion.

Version 1.1:
- Reproduccion con TextureView + MediaPlayer, descartada en v1.2 por arranque inestable en el movil.
- Selector Videos con miniaturas y orden por fecha, primero los mas recientes.
- Barra inferior de segmentos: I marca inicio, O anade segmento, E exporta todos los segmentos del video actual.

Los cortes se guardan como `nombre-LLC-00.00.00.000-00.00.10.000.mp4`.
Al reabrir la app se omiten los originales que ya tengan cortes `-LLC-`
o marcador `_MALO.txt`.
