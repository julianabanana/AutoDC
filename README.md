# 🏗️ AutoDC

An intelligent tool for designing 2D data centers with thermal and energy optimization, using the DeepSeek API and interactive visualization.

## ✨ Key Features

- 🖥️ Automatic layout generation with AI (DeepSeek API)
- 📊 Interactive 2D visualization with Matplotlib
- 🔥 Energy efficiency prediction model
- 🌐 Web interface with Streamlit
- 📏 Available space optimization

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/julianabanana/AutoDC.git
cd AutoDC
```

2. Install dependencies:

```bash

pip install matplotlib openai numpy scikit-learn streamlit

```
3. Configure your API key from the DeepSeek website:

```bash

client = OpenAI(api_key="Your-API-KEY", base_url="https://api.deepseek.com")
```
🛠️ How to Use

For the Streamlit web version, run from the console:
```bash

streamlit run AutoDC.py
```

Enter the requested data, and the execution result will open in your browser.
