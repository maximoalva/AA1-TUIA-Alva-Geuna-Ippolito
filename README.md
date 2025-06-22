# Predicción de lluvia en Australia ☔

Este repositorio contiene un modelo de *Machine Learning* que, a partir de variables climáticas históricas en distintas ciudades de Australia, permite predecir si lloverá o no al día siguiente.

## 🐳 Requisitos

Para poder ejecutar el modelo, es necesario tener instalado **Docker** en tu máquina. En caso de no tenerlo, podés instalarlo desde los siguientes enlaces:

- [Instalar Docker en Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Instalar Docker en Linux](https://docs.docker.com/desktop/setup/install/linux/)

---

## 🧪 Instrucciones de uso

A continuación se detallan los pasos necesarios para ejecutar el modelo:

### 1. Clonar el repositorio

Abrí la terminal y ejecutá:

```bash
git clone https://github.com/maximoalva/AA1-TUIA-Alva-Geuna-Ippolito
```

### 2. Cargar datos para realizar predicciones

Ubicá tu archivo `input.csv` con los datos para realizar predicciones en el directorio `files`.

📌 Asegurate de que el archivo tenga el formato adecuado requerido por el modelo. En el repositorio se incluye un ejemplo de archivo de entrada.

### 3. Construir la imagen de Docker

Desde la terminal, posicionado en la carpeta del directorio clonado, ejecutar:

```bash
docker build -t inference-tp2 ./docker
```

### 4. Ejecutar el contenedor

En caso de estar trabajando en **Windows**, reemplazá `RUTA` por la ruta absoluta de tu carpeta `files`.

En caso de estar trabajando en **Linux**, reemplazar `RUTA` por la ruta relativa a tu directorio `files` es suficiente.

Luego corré:

```bash
docker run -it --rm --name inference-tp2 -v RUTA:/files  inference-tp2
```

✅ Esto ejecutará automáticamente el modelo dentro del contenedor. Al finalizar, se generará un archivo `output.csv` en la carpeta `files` con las predicciones para cada fila de tu `input.csv`.

## 📁 Estructura esperada

```bash
.
├── files/
│   ├── input.csv                  # Archivo de entrada con datos climáticos
│   └── output.csv                 # Archivo generado automáticamente con las predicciones
├── docker/
│   ├── Dockerfile
│   ├── inference.py
│   ├── pipeline.pkl
│   ├── preprocessing.py
│   └── requirements.txt
├── README.md
└── TP2_clasificacion_AA1.ipynb
```
