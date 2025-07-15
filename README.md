# 🏗️ AutoDC

Una herramienta inteligente para diseñar data centers en 2D con optimización térmica y energética, utilizando la API de DeepSeek y visualización interactiva.

## ✨ Características Principales

- 🖥️ Generación automática de diseños con IA (DeepSeek API)
- 📊 Visualización 2D interactiva con Matplotlib
- 🔥 Modelo de predicción de eficiencia energética
- 🌐 Interfaz web con Streamlit
- 📏 Optimización del espacio disponible

## 🛠️ Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/julianabanana/AutoDC.git
cd AutoDC
```

2. Instala las dependencias:
```bash
pip install matplotlib openai numpy scikit-learn streamlit
```

3. Configura tu API key desde la página de deepseek:
```bash
 client = OpenAI(api_key="Tu-API-KEY", base_url="https://api.deepseek.com")
```

## 🛠️ Cómo usar
Para la versión web de streamlit, ejecuta desde consola:
```bash
streamlit run AutoDC.py
```

Igresas los datos que te pide, y se abriré en tu navegador el resultado de la ejecución.
