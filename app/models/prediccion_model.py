from sqlalchemy import Column, Integer, Float, String, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Prediccion(Base):
    __tablename__ = "predicciones"
    id = Column(Integer, primary_key=True)
    nombre_producto = Column(String(100))
    producto_id = Column(Integer)
    proyeccion_futura = Column(Float)
    ventas_pasadas = Column(Float)
    fecha_prediccion = Column(Date)
