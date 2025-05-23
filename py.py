from ultralytics import YOLO
import cv2
import mysql.connector
from datetime import datetime
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configuración de FastAPI
app = FastAPI()

# Configurar CORS para permitir solicitudes desde React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Dirección de tu app React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de la base de datos MySQL
db_config = {
    'host': 'localhost',
    'user': 'tu_usuario',  # Cambia por tu usuario de MySQL
    'password': 'tu_contraseña',  # Cambia por tu contraseña de MySQL
    'database': 'cellphone_db'
}

# Crear la carpeta para guardar imágenes si no existe
os.makedirs("imagenes", exist_ok=True)

# Conectar a la base de datos y crear la tabla si no existe
def init_db():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device VARCHAR(50),
            date_time DATETIME,
            image_path VARCHAR(255)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# Cargar el modelo YOLOv8
model = YOLO('yolov8s.pt')
confidence_threshold = 0.8
image_counter = 1  # Para nombrar imágenes como imagen1.jpg, imagen2.jpg, etc.

# Función para guardar en la base de datos
def save_detection(device, image_path):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    date_time = datetime.now()
    cursor.execute(
        "INSERT INTO detections (device, date_time, image_path) VALUES (%s, %s, %s)",
        (device, date_time, image_path)
    )
    conn.commit()
    cursor.close()
    conn.close()

# Endpoint para iniciar la detección
@app.get("/start_detection")
async def start_detection():
    global image_counter
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se puede acceder a la cámara.")

    while True:
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="No se pudo capturar el frame.")
        
        results = model(frame)
        for r in results[0].boxes:
            if r.cls == 67 and r.conf[0] > confidence_threshold:  # Solo celulares
                x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Celular {r.conf[0]:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Guardar imagen
                image_path = f"imagenes/imagen{image_counter}.jpg"
                cv2.imwrite(image_path, frame)
                
                # Guardar en la base de datos
                save_detection("Celular", image_path)
                image_counter += 1
        
        cv2.imshow("Detección de Celulares", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return {"message": "Detección finalizada"}

# Endpoint para obtener los datos de la base de datos
@app.get("/detections")
async def get_detections():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, device, date_time, image_path FROM detections")
    detections = cursor.fetchall()
    cursor.close()
    conn.close()
    return detections

# Endpoint para servir imágenes
@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"imagenes/{image_name}"
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Imagen no encontrada")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)