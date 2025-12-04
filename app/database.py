from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger(__name__)

# Datos de conexión (considera mover estos a variables de entorno)
DB_CONFIG = {
    "usuario": "root",
    "password": "DEquiJCu9)2003",
    "host": "localhost",
    "puerto": "3306",
    "base_de_datos": "demoinventario"
}

# Crear la cadena de conexión
DATABASE_URL = f"mysql+mysqlconnector://{DB_CONFIG['usuario']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['puerto']}/{DB_CONFIG['base_de_datos']}"

# Crear el motor de conexión
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)

# Session local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para modelos
Base = declarative_base()

def get_db_connection():
    """Obtiene una conexión a la base de datos"""
    try:
        conn = engine.connect()
        return conn
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise

def test_connection():
    """Prueba la conexión a la base de datos"""
    try:
        with engine.connect() as conn:
            resultado = conn.execute(text("SELECT NOW();"))
            for fila in resultado:
                logger.info(f"Conexión exitosa, fecha y hora del servidor: {fila[0]}")
                return True
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        return False

# Probar conexión al importar el módulo
if __name__ == "__main__":
    test_connection()