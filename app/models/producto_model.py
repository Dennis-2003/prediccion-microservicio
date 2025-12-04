# app/models/producto_model.py

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class Producto(Base):
    __tablename__ = "productos"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    descripcion = Column(String, nullable=True)
    precio_compra = Column(Float, nullable=True)
    precio_venta = Column(Float, nullable=True)
    stock = Column(Integer, nullable=True)
    stock_minimo = Column(Integer, nullable=True)
    categoria_id = Column(Integer, ForeignKey("categorias.id"), nullable=True)
    sede_id = Column(Integer, ForeignKey("sedes.id"), nullable=True)
    proveedor_id = Column(Integer, ForeignKey("proveedores.id"), nullable=True)

    # Relaciones opcionales
    # categoria = relationship("Categoria")
    # sede = relationship("Sede")
    # proveedor = relationship("Proveedor")
