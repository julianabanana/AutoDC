import matplotlib.pyplot as plt
import matplotlib.patches as patches
from openai import OpenAI
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st


# --- Parte 1: Generar diseño con DeepSeek ---
def generate_design(prompt):
    client = OpenAI(api_key="API-KEY", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user", 
                "content": prompt
            },
        ],
        stream=False
    )
    design = response.choices[0].message.content
    return design

def parse_design_response(response):
    """Convierte la respuesta de DeepSeek en listas de coordenadas"""
    racks = [(5,30,3,5), (15,30,3,5), (25,30,3,5), (35,30,3,5), (20,20,3,5)]  # Default 
    cooling = [(20,10,5,10)]
    electrical = [(20,5,3,10)]
    
    lines = response.strip().split('\n')
      
    if lines[0].rstrip() == "``` ":
        racks_count = int(lines[1].rstrip())
        racks_line = lines[2].rstrip().split(' ')
        cooling_count = int(lines[3].rstrip())
        cooling_line = lines[4].rstrip().split(' ')
        electrical_count = int(lines[5].rstrip())
        electrical_line = lines[6].rstrip().split(' ')
    else:
        racks_count = int(lines[0].rstrip())
        racks_line = lines[1].rstrip().split(' ')
        cooling_count = int(lines[2].rstrip())
        cooling_line = lines[3].rstrip().split(' ')
        electrical_count = int(lines[4].rstrip())
        electrical_line = lines[5].rstrip().split(' ')
        

    racks = []
    for rack in racks_line:
        racks += [list(int(num) for num in rack.split(','))]

    cooling = []            
    for cooler in cooling_line:
        cooling += [list(int(num) for num in cooler.split(','))]
        
    electrical = []
    for electric in electrical_line:
        electrical += [list(int(num) for num in electric.split(','))]
    
    return racks, cooling, electrical


# --- Parte 2: Visualización 2D ---
def plot_data_center(racks, cooling, electrical, xSize, ySize):
    """Dibuja el diseño del data center en 2D."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dibujar racks (azul)
    for r in racks:
        x, y , w, h = r
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=True, color="blue", label="Rack"))
    
    # Dibujar cooling (rojo)
    for cooler in cooling:
        x, y , w, h = cooler
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=True, color="red", label="Cooling"))
    
    # Dibujar área eléctrica (verde)
    for electric in electrical:
        x, y , w, h = electric
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=True, color="green", label="Electrical"))
    
    # Configuración del gráfico
    ax.set_xlim(0, int(xSize))
    ax.set_ylim(0, int(ySize))
    ax.set_aspect('equal')
    ax.legend(handles=[
        patches.Patch(color='blue', label='Racks'),
        patches.Patch(color='red', label='Cooling'),
        patches.Patch(color='green', label='Electrical')
    ])
    plt.title("Diseño de Data Center (2D)")
    plt.grid(True)
    plt.show()

#opción de visualización con streamlit
def plot_design(racks, cooling, electrical, xSize, ySize):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, int(xSize))
    ax.set_ylim(0, int(ySize))
    ax.grid(True)

    # Dibujar racks (rectángulos azules)
    for rack in racks:
        x, y , w, h = rack
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='blue', edgecolor='black', label='Rack'))

    # Dibujar cooling (rectángulos verdes)
    for cooler in cooling:
        x, y , w, h = cooler
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='green', edgecolor='black', label='Cooling'))

    # Dibujar electrical (rectángulos rojos)
    for electric in electrical:
        x, y , w, h = electric
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='red', edgecolor='black', label='Electrical'))

    ax.legend()
    st.pyplot(fig)

# --- Parte 3: Modelo de Eficiencia Térmica/Energética ---
def train_efficiency_model():
    """Entrena un modelo simple para predecir eficiencia."""
    # Datos sintéticos (ejemplo: [área_racks, distancia_cooling] -> eficiencia_termica)
    X = np.array([[5, 2], [10, 5], [15, 1], [8, 3], [12, 4]])
    y = np.array([0.9, 0.6, 0.8, 0.7, 0.5])  # Eficiencia (0-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Modelo entrenado. Score: {model.score(X_test, y_test):.2f}")
    return model

def predict_efficiency(model, racks, cooling):
    """Predice la eficiencia térmica del diseño actual."""
    total_rack_area = sum(w * h for (_, _, w, h) in racks)
    avg_distance = np.mean([np.sqrt((x - cooling[0][0])**2 + (y - cooling[0][1])**2) for (x, y, _, _) in racks])
    efficiency = model.predict([[total_rack_area, avg_distance]])[0]
    st.write(f"Eficiencia energética predicha: {efficiency:.2f} (1=óptima)")

# --- Flujo Principal ---
if __name__ == "__main__":
    # Paso 1: Generar diseño
    xSize = input("Ingrese el ancho del área disponible para el diseño:")
    ySize = input("Ingrese el largo del área disponible para el diseño:")
    user_prompt = input("Describa el diseño:")+ "Debes realizar este diseño en un espacio de "+ xSize + "x" + ySize + ". Prioriza el uso del espacio y el ahorro de energía. No añadas texto alguno. No puedes poner un rack sobre otro, todos tienen que verse claramente al graficar las coordenadas. Tu output debe contener 6 lineas. La primera línea debe contener la cantidad de racks que tiene el diseño. La segunda linea debe contener las coordendas de los racks, cada coordenada tiene el formato posicion eje x,posocion eje y,alto,ancho. Deben estar separados por un espacio. La tercera línea debe contener la cantidad de cooling que hay. La cuartra línea debe contener las coordendas de los cooling, cada coordenada tiene el formato posicion eje x,posocion eje y,alto,ancho. Deben estar separados por un espacio. La quinta línea contiene la cantidad de electrical que hay. La sexta línea debe contener las coordendas de los electrical, cada coordenada tiene el formato posicion eje x,posocion eje y,alto,ancho. Deben estar separados por un espacio. No añadas texto al final"
    
    design_response = generate_design(user_prompt)
    
    # Paso 2: Parsear y visualizar (usando valores por defecto por simplicidad)
    racks, cooling, electrical = parse_design_response(design_response)
    
    st.title("Diseño de Data Center 2D")
    st.write("Este es el resultado de la API de DeepSeek del diseño en 2d para el prompt dado. Para más información, consultar en https://github.com/julianabanana/AutoDC")
    
    plot_data_center(racks, cooling, electrical, xSize, ySize)
    plot_design(racks, cooling, electrical, xSize, ySize)
    
    # Paso 3: Entrenar y predecir eficiencia
    efficiency_model = train_efficiency_model()
    predict_efficiency(efficiency_model, racks, cooling)
    
    
    
    
