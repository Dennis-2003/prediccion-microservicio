from flask import Flask
from flask_cors import CORS
import os
import mysql.connector
import json
import numpy as np
from datetime import datetime, date

class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder personalizado que maneja tipos numpy y datetime"""
    def default(self, obj):
        # Manejar tipos numpy
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # Llamar al método padre para otros tipos
        return super().default(obj)

def create_app():
    app = Flask(__name__)
    
    # AGREGAR EL ENCODER PERSONALIZADO (ESENCIAL)
    app.json_encoder = CustomJSONEncoder

    # CONFIGURACIÓN CORS CON EL NOMBRE REAL DE LA MÁQUINA
    CORS(app, origins=[
        "http://localhost:9090",
        "http://127.0.0.1:9090",
        "http://segundo:9090",
        "http://segundo:8080"
    ])

    # Configuración de MySQL (local)
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "DEquiJCu9)2003")
    DB_NAME = os.getenv("DB_NAME", "demoinventario")
                        
    # Intentar conexión para verificar
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print("Conexión exitosa a MySQL")
        connection.close()
    except mysql.connector.Error as err:
        print("Error conectando a MySQL:", err)

    # IMPORTACIÓN DENTRO DE LA FUNCIÓN PARA EVITAR CIRCULAR IMPORTS
    from app.routes import bp as predicciones_bp
    app.register_blueprint(predicciones_bp)

    # Imprimir rutas para debug
    print(" Rutas registradas:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)