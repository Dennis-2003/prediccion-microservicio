import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.database import get_db_connection
import logging

logger = logging.getLogger(__name__)

class SistemaExpertoService:
    
    def __init__(self):
        self.reglas_abastecimiento = self._cargar_reglas_abastecimiento()
        self.umbrales_alertas = self._cargar_umbrales_alertas()
        
    def _cargar_reglas_abastecimiento(self):
        """Carga reglas  para minimarket"""
        return {
            'alta_demanda': {'min_ventas': 8, 'dias_stock': 7, 'factor_seguridad': 1.2},
            'media_demanda': {'min_ventas': 3, 'dias_stock': 14, 'factor_seguridad': 1.1},
            'baja_demanda': {'min_ventas': 1, 'dias_stock': 21, 'factor_seguridad': 1.05},
            'critico': {'min_ventas': 0, 'dias_stock': 30, 'factor_seguridad': 1.0}
        }
    
    def _cargar_umbrales_alertas(self):
        """Carga REALISTAS"""
        return {
            'vencimiento_proximo': 15,
            'stock_minimo': 5,
            'rotacion_baja': 0.1,
            'sobrestock': 40
        }
    
    def obtener_datos_productos_completos(self):
        """Obtiene datos REALES de productos por sede"""
        try:
            conn = get_db_connection()
            
            # ventas por sede
            query = """
            SELECT 
                p.id as producto_id, 
                p.nombre,
                COALESCE(p.stock, 10) as stock_actual,
                COALESCE(p.stock_minimo, 5) as stock_minimo,
                COALESCE(p.precio_compra, 0) as precio_compra,
                COALESCE(p.precio_venta, 0) as precio_venta,
                -- Ventas por sede (más realista)
                (
                    SELECT COALESCE(SUM(dv.cantidad), 0) 
                    FROM detalles_venta dv 
                    JOIN ventas v ON dv.venta_id = v.id 
                    WHERE dv.producto_id = p.id 
                    AND v.sede_id = 1
                    AND v.fecha >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                ) as ventas_sede_1,
                (
                    SELECT COALESCE(SUM(dv.cantidad), 0) 
                    FROM detalles_venta dv 
                    JOIN ventas v ON dv.venta_id = v.id 
                    WHERE dv.producto_id = p.id 
                    AND v.sede_id = 2
                    AND v.fecha >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                ) as ventas_sede_2,
                COALESCE(p.activo, 1) as activo
            FROM productos p 
            WHERE p.activo = 1 OR p.activo IS NULL
            ORDER BY p.id
            """
            
            productos_df = pd.read_sql(query, conn)
            conn.close()
            
            logger.info(f" Datos por sede: {len(productos_df)} productos")
            
            if productos_df.empty:
                logger.warning(" No hay productos activos")
                return pd.DataFrame()
                
            return productos_df
            
        except Exception as e:
            logger.error(f" Error en consulta productos: {e}")
            return pd.DataFrame()

    def generar_recomendaciones_reabastecimiento(self, predicciones_ventas):
        """Genera recomendaciones  usando datos por sede"""
        start_time = datetime.now()
        
        try:
            if not predicciones_ventas or not predicciones_ventas.get("sedes"):
                logger.warning("No hay datos de predicciones")
                return []
            
            # Obtener datos REALES de productos
            productos_df = self.obtener_datos_productos_completos()
            
            logger.info(f" Procesando {len(predicciones_ventas.get('sedes', []))} sedes")
            logger.info(f" Productos en BD: {len(productos_df)}")
            
            recomendaciones = []
            productos_procesados = 0
            productos_con_datos_reales = 0
            
            for sede_pred in predicciones_ventas.get("sedes", []):
                sede_id = sede_pred.get("sede_id")
                sede_nombre = sede_pred.get("sede_nombre", "Sede")
                
                for producto_pred in sede_pred.get("productos", []):
                    producto_id = producto_pred.get("producto_id")
                    productos_procesados += 1
                    
                    try:
                        # Buscar producto en BD REAL
                        producto_match = productos_df[productos_df['producto_id'] == producto_id]
                        
                        if producto_match.empty:
                            # Si no está en BD, usar solo predicciones
                            recomendacion = self._crear_recomendacion_solo_predicciones(
                                sede_id, sede_nombre, producto_pred
                            )
                            recomendacion['fuente_datos'] = 'PREDICCIONES_SOLO'
                        else:
                            # Procesar con datos REALES de BD por sede
                            producto_row = producto_match.iloc[0]
                            recomendacion = self._procesar_con_datos_reales_por_sede(
                                sede_id, sede_nombre, producto_pred, producto_row
                            )
                            productos_con_datos_reales += 1
                            recomendacion['fuente_datos'] = 'BD_REAL'
                        
                        if recomendacion:
                            recomendaciones.append(recomendacion)
                            
                    except Exception as e:
                        logger.error(f"Error producto {producto_id}: {e}")
                        continue
            
            # Métricas REALES
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"""
            - Tiempo procesamiento: {execution_time:.2f}s
            - Total productos procesados: {productos_procesados}
            - Con datos BD reales: {productos_con_datos_reales}
            - Recomendaciones generadas: {len(recomendaciones)}
            - Efectividad: {(productos_con_datos_reales / max(1, productos_procesados)) * 100:.1f}%
            """)
            
            return sorted(recomendaciones, key=lambda x: x['prioridad'], reverse=True)
            
        except Exception as e:
            logger.error(f" Error crítico en sistema experto: {e}")
            return []

    def _procesar_con_datos_reales_por_sede(self, sede_id, sede_nombre, producto_pred, producto_row):
        """Procesa producto con datos  por sede -"""
        try:
            # Datos REALES de productos
            stock_actual = self._safe_float(producto_row['stock_actual'])
            stock_minimo = self._safe_float(producto_row['stock_minimo'])
            precio_compra = self._safe_float(producto_row['precio_compra'])
            
            # Obtener ventas REALES por sede
            if sede_id == 1:
                ventas_reales_30d = self._safe_float(producto_row['ventas_sede_1'])
            else:
                ventas_reales_30d = self._safe_float(producto_row['ventas_sede_2'])
            
            # Calcular rotación REALISTA por sede
            ventas_promedio = ventas_reales_30d / 30.0
            
            # LIMITAR a valores REALISTAS para minimarket
            ventas_promedio = min(ventas_promedio, 15)  
            
            # Si no hay ventas reales, usar un mínimo realista
            if ventas_promedio < 0.3:
                ventas_promedio = 0.5  # Mínimo realista de 0.5 uds/día
            
            # Combinar con predicciones
            proyeccion_diaria = producto_pred.get("proyeccion_diaria", [])
            if proyeccion_diaria:
                predicciones = [
                    min(pdia.get("prediccion", 0), 10)  # Límite realista por día
                    for pdia in proyeccion_diaria 
                    if isinstance(pdia, dict)
                ]
                demanda_predicha = np.mean(predicciones) if predicciones else ventas_promedio
            else:
                demanda_predicha = ventas_promedio
            
            # histórico  con predicción (70% histórico, 30% predicción)
            demanda_esperada = (ventas_promedio * 0.7) + (demanda_predicha * 0.3)
            demanda_esperada = max(demanda_esperada, 0.5)  # Mínimo 
            demanda_esperada = min(demanda_esperada, 12)   # Máximo 
            
            # Clasificar demanda 
            if demanda_esperada >= 8:
                regla = 'alta_demanda'
            elif demanda_esperada >= 3:
                regla = 'media_demanda'
            elif demanda_esperada >= 1:
                regla = 'baja_demanda'
            else:
                regla = 'critico'
            
            # Calcular cantidades
            dias_cobertura = self.reglas_abastecimiento[regla]['dias_stock']
            factor_seguridad = self.reglas_abastecimiento[regla]['factor_seguridad']
            
            # Stock recomendado 
            stock_recomendado = demanda_esperada * dias_cobertura * factor_seguridad
            pedido_recomendado = max(0, stock_recomendado - stock_actual)
            
            # Límites más conservadores
            limites_pedido = {
                'alta_demanda': 20,   # Máximo 20 unidades
                'media_demanda': 12,  # Máximo 12 unidades  
                'baja_demanda': 6,    # Máximo 6 unidades
                'critico': 3          # Máximo 3 unidades
            }
            
            limite = limites_pedido.get(regla, 8)
            pedido_recomendado = min(pedido_recomendado, limite)
            stock_recomendado = min(stock_recomendado, 50)  # Stock máximo 
            
            # Asegurar pedido mínimo
            if pedido_recomendado > 0:
                pedido_recomendado = max(pedido_recomendado, 2)  # Mínimo 
            
            # Calcular días de stock 
            dias_stock_actual = stock_actual / demanda_esperada if demanda_esperada > 0 else 999
            
            # Generar alertas 
            alertas = self._generar_alertas_reales_corregidas(
                producto_pred.get("producto_id"), stock_actual, stock_minimo, 
                demanda_esperada, ventas_promedio, producto_pred.get("nombre_producto", "Producto")
            )
            
            return {
                'sede_id': sede_id,
                'sede_nombre': sede_nombre,
                'producto_id': producto_pred.get("producto_id"),
                'nombre_producto': producto_pred.get("nombre_producto", producto_row['nombre']),
                'categoria_demanda': regla.upper(),
                'stock_actual': int(stock_actual),
                'stock_minimo': int(stock_minimo),
                'stock_recomendado': int(round(stock_recomendado)),
                'pedido_recomendado': int(round(pedido_recomendado)),
                'dias_cobertura_actual': round(dias_stock_actual, 1),
                'dias_cobertura_recomendado': dias_cobertura,
                'prioridad': self._calcular_prioridad_real(regla, alertas),
                'alertas': alertas,
                'costo_pedido_estimado': round(pedido_recomendado * precio_compra, 2),
                'justificacion': self._generar_justificacion_real_corregida(regla, demanda_esperada, ventas_promedio),
                'ventas_reales_30d': int(ventas_reales_30d),
                'demanda_promedio_real': round(ventas_promedio, 2),
                'demanda_esperada_calculada': round(demanda_esperada, 2)
            }
            
        except Exception as e:
            logger.error(f"Error procesando producto con datos reales: {e}")
            return None

    def _generar_alertas_reales_corregidas(self, producto_id, stock_actual, stock_minimo, demanda_esperada, ventas_promedio, nombre_producto):
        """Genera alertas REALISTAS corregidas"""
        alertas = []
        
        # Alertas de stock
        if stock_actual <= 1:  
            alertas.append({
                'tipo': 'STOCK_CRITICO',
                'mensaje': f'Stock CRÍTICO: {stock_actual} unidad(es)',
                'severidad': 'ALTA',
                'accion': 'REABASTECER URGENTE'
            })
        elif stock_actual <= stock_minimo:
            alertas.append({
                'tipo': 'STOCK_BAJO', 
                'mensaje': f'Stock bajo: {stock_actual} unidades (mínimo: {stock_minimo})',
                'severidad': 'MEDIA',
                'accion': 'REABASTECER PRONTO'
            })
        
        # Alerta de sobrestock 
        if stock_actual > 30:  # Sobrestock  para minimarket
            alertas.append({
                'tipo': 'SOBRESTOCK',
                'mensaje': f'Stock alto: {stock_actual} unidades',
                'severidad': 'MEDIA', 
                'accion': 'CONGELAR PEDIDOS'
            })
        
        # Alerta de baja 
        if ventas_promedio < 0.3:  # Menos de 1 unidad cada 3 días
            alertas.append({
                'tipo': 'BAJA_ROTACION',
                'mensaje': f'Rotación baja: {ventas_promedio:.1f} uds/día',
                'severidad': 'MEDIA',
                'accion': 'REVISAR PRODUCTO'
            })
        
        return alertas

    def _generar_justificacion_real_corregida(self, categoria, demanda_esperada, ventas_promedio):
        """Genera justificación REALISTA corregida"""
        justificaciones = {
            'alta_demanda': f"Demanda alta ({demanda_esperada:.1f} uds/día) - stock para 1 semana",
            'media_demanda': f"Demanda media ({demanda_esperada:.1f} uds/día) - stock para 2 semanas", 
            'baja_demanda': f"Demanda baja ({demanda_esperada:.1f} uds/día) - stock mínimo necesario",
            'critico': f"Demanda muy baja ({demanda_esperada:.1f} uds/día) - evaluar continuidad"
        }
        return justificaciones.get(categoria, f"Demanda: {demanda_esperada:.1f} uds/día")

    def _crear_recomendacion_solo_predicciones(self, sede_id, sede_nombre, producto_pred):
        """Crea recomendación cuando solo hay predicciones"""
        ventas_pasadas = producto_pred.get("ventas_pasadas", 0)
        ventas_promedio = max(0, ventas_pasadas) / 30.0
        
        # Valores más realistas
        if ventas_promedio >= 8:
            regla = 'alta_demanda'
            pedido_recomendado = 12
        elif ventas_promedio >= 3:
            regla = 'media_demanda' 
            pedido_recomendado = 8
        else:
            regla = 'baja_demanda'
            pedido_recomendado = 4
        
        return {
            'sede_id': sede_id,
            'sede_nombre': sede_nombre,
            'producto_id': producto_pred.get("producto_id"),
            'nombre_producto': producto_pred.get("nombre_producto", "Producto"),
            'categoria_demanda': regla.upper(),
            'stock_actual': 0,
            'stock_minimo': 5,
            'stock_recomendado': pedido_recomendado + 8,
            'pedido_recomendado': pedido_recomendado,
            'dias_cobertura_actual': 0,
            'dias_cobertura_recomendado': 10,
            'prioridad': 4,
            'alertas': [{
                'tipo': 'DATOS_LIMITADOS',
                'mensaje': 'Información de inventario no disponible',
                'severidad': 'MEDIA', 
                'accion': 'VERIFICAR EN SISTEMA'
            }],
            'costo_pedido_estimado': 0,
            'justificacion': f'Basado en ventas históricas ({ventas_promedio:.1f} uds/día)',
            'ventas_reales_30d': 0,
            'demanda_promedio_real': 0,
            'demanda_esperada_calculada': round(ventas_promedio, 2)
        }

    def _calcular_prioridad_real(self, categoria_demanda, alertas):
        prioridad_base = {'alta_demanda': 7, 'media_demanda': 5, 'baja_demanda': 3, 'critico': 6}
        prioridad = prioridad_base.get(categoria_demanda, 4)
        
        for alerta in alertas:
            if alerta.get('severidad') == 'ALTA':
                prioridad += 3
            elif alerta.get('severidad') == 'MEDIA':
                prioridad += 1
        
        return max(1, min(prioridad, 10))

    def _safe_float(self, value, default=0.0):
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def obtener_estadisticas_sistema(self):
        return {
            'reglas_abastecimiento': self.reglas_abastecimiento,
            'umbrales_alertas': self.umbrales_alertas,
            'fecha_actualizacion': datetime.now().isoformat(),
            'version': '6.0-realista-corregido',
            'descripcion': 'Sistema experto - datos por sede'
        }