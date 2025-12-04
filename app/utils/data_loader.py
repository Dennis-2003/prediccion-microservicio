from app.database import get_db_connection
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def cargar_ventas(ultimos_meses=6):
    """Carga ventas usando conexión directa"""
    try:
        conn = get_db_connection()
        fecha_limite = (datetime.now() - timedelta(days=ultimos_meses*30)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT id as venta_id, fecha, cliente_id, sede_id, usuario_id, total
        FROM ventas 
        WHERE fecha >= '{fecha_limite}'
        """
        
        ventas_df = pd.read_sql(query, conn)
        conn.close()
        
        # Convertir fecha a string
        if 'fecha' in ventas_df.columns:
            ventas_df['fecha'] = ventas_df['fecha'].astype(str)
        
        logger.info(f"Cargadas {len(ventas_df)} ventas")
        return ventas_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error cargando ventas: {e}")
        return []

def cargar_detalles(ultimos_meses=6):
    """Carga detalles usando conexión directa"""
    try:
        conn = get_db_connection()
        fecha_limite = (datetime.now() - timedelta(days=ultimos_meses*30)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT dv.id as detalle_id, dv.venta_id, dv.producto_id, 
               dv.cantidad, dv.precio_unitario, dv.subtotal
        FROM detalles_venta dv
        JOIN ventas v ON dv.venta_id = v.id
        WHERE v.fecha >= '{fecha_limite}'
        """
        
        detalles_df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Cargados {len(detalles_df)} detalles de venta")
        return detalles_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error cargando detalles: {e}")
        return []

def cargar_productos():
    """Carga productos usando la estructura real de la BD"""
    try:
        conn = get_db_connection()
        
        # Query con las columnas reales de la BD
        query = """
        SELECT 
            id as producto_id,
            COALESCE(nombre, 'Producto Sin Nombre') as nombre,
            COALESCE(descripcion, '') as descripcion,
            COALESCE(precio_compra, 0.0) as precio_compra,
            COALESCE(precio_venta, 0.0) as precio_venta,
            COALESCE(stock, 0) as stock_actual,
            COALESCE(stock_minimo, 0) as stock_minimo,
            COALESCE(categoria_id, 1) as categoria_id,
            COALESCE(sede_id, 1) as sede_id,
            COALESCE(proveedor_id, 1) as proveedor_id
        FROM productos
        WHERE activo = 1 OR activo IS NULL
        """
        
        productos_df = pd.read_sql(query, conn)
        conn.close()
        
        # Procesar los resultados
        resultado = []
        for _, row in productos_df.iterrows():
            producto = {
                'producto_id': int(row['producto_id']),
                'nombre': str(row['nombre']),
                'descripcion': str(row['descripcion']),
                'precio_compra': float(row['precio_compra']),
                'precio_venta': float(row['precio_venta']),
                'stock_actual': float(row['stock_actual']),
                'stock_minimo': float(row['stock_minimo']),
                'categoria_id': int(row['categoria_id']) if pd.notna(row['categoria_id']) else 1,
                'sede_id': int(row['sede_id']) if pd.notna(row['sede_id']) else 1,
                'proveedor_id': int(row['proveedor_id']) if pd.notna(row['proveedor_id']) else 1
            }
            resultado.append(producto)
        
        logger.info(f" Productos cargados: {len(resultado)}")
        return resultado
        
    except Exception as e:
        logger.error(f"Error cargando productos: {e}")
        return []

# Exportar las funciones
__all__ = ['cargar_ventas', 'cargar_detalles', 'cargar_productos']