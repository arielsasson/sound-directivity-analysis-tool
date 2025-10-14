# Análisis de Directividad

Una herramienta profesional para el análisis de patrones de directividad de audio utilizando gráficos de globo (balloon plots) y visualizaciones polares.

## Estructura del Proyecto

El proyecto ha sido refactorizado siguiendo mejores prácticas de desarrollo:

```
src/
├── analisis_directividad/
│   ├── __init__.py
│   ├── config.py                 # Configuraciones y constantes
│   ├── audio/                    # Procesamiento de audio
│   │   ├── __init__.py
│   │   └── processing.py
│   ├── data/                     # Procesamiento de datos
│   │   ├── __init__.py
│   │   └── processing.py
│   ├── visualization/            # Visualizaciones
│   │   ├── __init__.py
│   │   └── plots.py
│   ├── gui/                      # Interfaz gráfica
│   │   ├── __init__.py
│   │   ├── widgets.py
│   │   ├── main_window.py
│   │   └── handlers.py
│   └── utils/                    # Utilidades
│       ├── __init__.py
│       └── data_generation.py
├── main.py                       # Punto de entrada principal
└── requirements.txt              # Dependencias
```

## Características

- **Procesamiento de Audio**: Lectura de archivos WAV, filtrado de bandas de frecuencia, cálculo de SPL
- **Análisis de Datos**: Normalización, promediado de mediciones redundantes, interpolación
- **Visualización 3D**: Gráficos de globo interactivos usando VTK/Vedo
- **Visualización Polar**: Gráficos polares 2D para diferentes vistas (superior, frontal, sagital)
- **Interfaz Gráfica**: GUI intuitiva desarrollada con PyQt5

## Instalación

1. Clona el repositorio
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta la aplicación:

```bash
python main.py
```

### Flujo de Trabajo

1. **Importar Mediciones**: Selecciona la carpeta que contiene los archivos de medición
2. **Configurar Parámetros**: 
   - Tipo de filtrado (Tercios de octava u Octava)
   - Rangos de elevación y azimut
   - Opciones de interpolación
3. **Calcular**: Procesa las mediciones y genera los datos de directividad
4. **Visualizar**: Explora los resultados usando los gráficos 3D y polares
5. **Exportar**: Guarda los datos procesados en formato CSV

## Módulos Principales

### Audio Processing (`audio/`)
- Lectura y procesamiento de archivos de audio
- Filtrado de bandas de frecuencia
- Cálculo de niveles SPL y calibración

### Data Processing (`data/`)
- Análisis estadístico de datos SPL
- Normalización de patrones de directividad
- Procesamiento de mediciones redundantes

### Visualization (`visualization/`)
- Creación de gráficos de globo 3D
- Visualizaciones polares 2D
- Conversión de coordenadas esféricas a cartesianas

### GUI (`gui/`)
- Interfaz de usuario principal
- Widgets personalizados
- Manejo de eventos y lógica de negocio

### Utils (`utils/`)
- Generación de datos de balloon
- Funciones de interpolación
- Utilidades comunes

## Configuración

Las constantes y configuraciones se encuentran en `config.py`:

- Bandas de frecuencia predefinidas
- Rangos de ángulos válidos
- Configuraciones de la GUI
- Parámetros de visualización

## Contribución

El código ha sido estructurado siguiendo principios de:

- **Separación de responsabilidades**: Cada módulo tiene una función específica
- **Reutilización**: Funciones modulares y bien documentadas
- **Mantenibilidad**: Código limpio y bien organizado
- **Extensibilidad**: Fácil agregar nuevas funcionalidades

## Licencia

Este proyecto está desarrollado para análisis académico de directividad de audio.
