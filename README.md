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

#Prompt example
"Design a small data center with an area of 20x10m, including 2 racks of servers, on area for electrical management and one area for cooling."
<img width="729" height="878" alt="Screenshot 2025-09-29 at 10 11 48 PM" src="https://github.com/user-attachments/assets/d61455be-b767-4e7d-846b-0ae32d08924e" />

