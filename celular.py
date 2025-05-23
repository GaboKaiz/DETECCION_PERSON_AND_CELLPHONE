from ultralytics import YOLO
import cv2

# Cargar el modelo preentrenado de YOLOv8
model = YOLO('yolov8s.pt')  # Asegúrate de tener el modelo adecuado para detectar personas y celulares

# Umbral de confianza (puedes ajustarlo según tu necesidad)
confidence_threshold = 0.8  # Detecta solo con confianza mayor al 90%

# Función para realizar detección en tiempo real desde la cámara (solo celular y persona)
def detect_cellphone_person():
    # Abrir la cámara (0 por defecto es la cámara principal de tu PC)
    cap = cv2.VideoCapture(0)
    
    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return
    
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        
        if not ret:
            print("No se pudo capturar el frame.")
            break
        
        # Realizar la inferencia de detección de objetos en cada cuadro del video
        results = model(frame)
        
        # Filtrar solo los resultados de celulares (67) y personas (0)
        filtered_results = []
        for r in results[0].boxes:
            # Si la clase es 0 (persona) o 67 (celular) y la confianza es mayor al umbral
            if (r.cls == 0 or r.cls == 67) and r.conf[0] > confidence_threshold:
                filtered_results.append(r)

        # Si se detectan personas o celulares, dibujar las cajas
        for result in filtered_results:
            # Dibujar cajas manualmente (opcionalmente puedes personalizar color, grosor, etc.)
            x1, y1, x2, y2 = result.xyxy[0].tolist()  # Obtener las coordenadas de la caja
            label = model.names[int(result.cls)]  # Obtener el nombre de la clase (persona o celular)
            confidence = result.conf[0].item()  # Obtener la confianza

            # Dibujar la caja con OpenCV
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar el cuadro con las cajas de detección de celulares y personas
        cv2.imshow("Detección de Celulares y Personas", frame)

        # Esperar 1 ms y salir si presionas la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()

# Llamar la función para ejecutar detección de celulares y personas
detect_cellphone_person()
