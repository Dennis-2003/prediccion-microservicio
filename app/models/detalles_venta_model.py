from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class DetallesVenta(Base):
    __tablename__ = "detalles_venta"

    id = Column(Integer, primary_key=True, index=True)
    cantidad = Column(Integer)
    precio_unitario = Column(Float)
    subtotal = Column(Float)
    producto_id = Column(Integer, ForeignKey("productos.id")) 
    venta_id = Column(Integer, ForeignKey("ventas.id"))

    # Agregar relaciones 
    producto = relationship("Producto", back_populates="detalles_venta")
    venta = relationship("Venta", back_populates="detalles")