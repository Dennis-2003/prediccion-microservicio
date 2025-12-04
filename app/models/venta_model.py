from sqlalchemy import Column, Integer, Float, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Venta(Base):
    __tablename__ = "ventas"

    id = Column(Integer, primary_key=True)
    fecha = Column(Date)
    total = Column(Float)
    cliente_id = Column(Integer)
    sede_id = Column(Integer)
    usuario_id = Column(Integer)
