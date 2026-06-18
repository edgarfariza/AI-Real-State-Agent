# AI Real Estate Agent
# Predicción de precios de viviendas con Amazon SageMaker

Este proyecto forma parte de mi formación en Cloud Computing. El objetivo inicial era utilizar Amazon SageMaker Canvas, una herramienta visual que permite crear modelos de Inteligencia Artificial sin escribir código, para adivinar el precio de las viviendas en función de sus características (como el número de habitaciones o la zona donde se encuentran). 

El propósito de este repositorio es explicar los problemas que encontré con los permisos de la cuenta de AWS, cómo los solucioné cambiando de estrategia y el código que utilicé para crear el modelo definitivo.

---

## Servicios de la Nube Utilizados

Para realizar este proyecto he trabajado en la nube de Amazon Web Services (AWS) utilizando las siguientes herramientas:

* **Amazon SageMaker:** La plataforma principal de AWS para trabajar con Inteligencia Artificial y modelos predictivos.
* **Instancias de Cuaderno (Notebook Instances):** Servidores virtuales dedicados que permiten abrir el entorno JupyterLab para escribir e instalar scripts en Python de forma aislada y segura.
* **Amazon S3:** El servicio de almacenamiento en la nube donde se guardan los archivos de datos utilizados.

---

## El Problema de Negocio (Sector Inmobiliario)

El ejercicio se basa en un caso real del sector inmobiliario. Contamos con una tabla de datos (dataset) que incluye información de muchas viviendas: su ubicación por coordenadas, la antigüedad del edificio, el número de habitaciones, los dormitorios y el nivel económico de los vecinos de la zona. 

El objetivo es entrenar al sistema para que aprenda la relación que hay entre todos esos datos y sea capaz de calcular, de forma automática, el precio de una casa. 

### Utilidad para una Empresa:
Tener un modelo que calcule los precios de forma automática permite a una inmobiliaria:
* Tasar los pisos mucho más rápido nada más entrar en la cartera.
* Encontrar chollos o casas que están a la venta por debajo del precio real del mercado.
* Poner precios de venta basados en estadísticas reales y no en una simple intuición.

---

## Desarrollo Paso a Paso

### 1. Bloqueo de Permisos y Cambio de Estrategia
Al intentar abrir la herramienta visual SageMaker Canvas dentro del laboratorio de la universidad, el sistema me denegó el acceso con un error de tipo `AccessDeniedException`. Esto ocurrió porque las cuentas de estudiante vienen muy limitadas de fábrica y no tienen permisos para activar ciertas opciones de usuario en AWS (como el Identity Center o el Resource Access Manager).

Como no podía cambiar los permisos de la cuenta, decidí solucionar el problema como programador: en lugar de usar la interfaz visual sin código, creé una **Instancia de Cuaderno tradicional (Notebook Instance)** de tipo `ml.t3.medium` usando el rol con los permisos que ya venían configurados en el laboratorio (`LabRole`). Esta vía trabaja de forma independiente, esquiva el bloqueo y me permitió continuar con el proyecto escribiendo el código yo mismo.

![Instancia de Cuaderno Activa](img/01-instancia-cuaderno.png)

### 2. Carga de Datos y Gráfica de Relaciones
Una vez dentro de JupyterLab, subí el archivo `housing.csv` del taller, abrí un archivo nuevo de Python 3 y cargué la tabla con los datos. Como la máquina venía totalmente limpia de fábrica, lo primero que tuve que hacer fue instalar las librerías necesarias ejecutando los comandos `!pip install seaborn` y `!pip install scikit-learn` directamente en las celdas.

Para entender mejor la información antes de entrenar al sistema, pinté un mapa de calor. Esta gráfica muestra de forma visual qué características influyen más en que una casa sea más cara o más barata. Al mirar los resultados, se ve claro que la columna `median_income` (el sueldo de la zona) es la que más fuerza tiene a la hora de subir el precio de la vivienda (`median_house_value`), con un valor de 0.69.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargamos el archivo CSV oficial del taller de AWS
df = pd.read_csv('housing.csv')

# 2. Mostramos en pantalla las primeras 3 filas para comprobar los datos
print("Primeras filas del archivo del taller:")
print(df.head(3))

# 3. Dibujamos el mapa de calor usando solo las columnas numéricas
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Dark2', fmt='.2f')
plt.title('Matriz de Correlación - Dataset Oficial de AWS')
plt.show()
