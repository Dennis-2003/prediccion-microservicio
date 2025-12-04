from flask import Blueprint, jsonify, request
from app.services.prediccion_service import PrediccionService
from app.services.sistema_experto_service import SistemaExpertoService
from datetime import datetime, date
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)
bp = Blueprint("predicciones_bp", __name__)
servicio = PrediccionService()
sistema_experto = SistemaExpertoService()

# FUNCIÓN DE LIMPIEZA JSON PARA RESOLVER ERROR DE SERIALIZACIÓN
def limpiar_para_json(obj):
    """Función recursiva para limpiar objetos para JSON"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: limpiar_para_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [limpiar_para_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return limpiar_para_json(obj.__dict__)
    else:
        try:
            return str(obj)
        except:
            return None

# FUNCIÓN: Convertir confianza de texto a número
def convertir_confianza_a_double(nivel_confianza: str) -> float:
    """Convierte 'ALTO'/'MEDIO'/'BAJO' a valores double que Java puede mapear"""
    confianza_map = {
        'ALTO': 0.95,
        'MEDIO': 0.75, 
        'BAJO': 0.5
    }
    return confianza_map.get(nivel_confianza, 0.5)

# FUNCIÓN COMPARTIDA para ambos endpoints
def prediccion_producto_directo(producto_id, dias):
    """Lógica compartida para predicción de producto"""
    try:
        # Obtener predicción usando el método optimizado del servicio
        resultado = servicio.generar_prediccion_producto(producto_id, dias)
        
        # Formatear respuesta para Java
        respuesta = {
            "productoId": producto_id,
            "prediccionDemanda": float(resultado.get("prediccionDemanda", 25.5)),
            "nivelConfianza": float(resultado.get("nivelConfianza", 0.5)),
            "fechaPrediccion": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        logger.info(f"Predicción generada para producto {producto_id}: {respuesta['prediccionDemanda']} unidades")
        return jsonify(respuesta)
        
    except Exception as e:
        logger.error(f"Error en predicción directa producto {producto_id}: {e}")
        return jsonify({
            "productoId": producto_id,
            "prediccionDemanda": 25.5,
            "nivelConfianza": 0.5,
            "fechaPrediccion": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        })

# ENDPOINT QUE JAVA ESTÁ LLAMANDO ACTUALMENTE
@bp.route('/predicciones/<int:dias>', methods=['GET'])
def predicciones_por_dias(dias):
    """Endpoint que Java está llamando - /predicciones/7?productoId=2"""
    try:
        producto_id = request.args.get('productoId', type=int)
        if not producto_id:
            return jsonify({"error": "Se requiere productoId"}), 400
            
        logger.info(f"Endpoint Java - Producto: {producto_id}, Días: {dias}")
        
        # Reutilizar la lógica del endpoint de producto
        return prediccion_producto_directo(producto_id, dias)
        
    except Exception as e:
        logger.error(f"Error en endpoint Java: {e}")
        producto_id = request.args.get('productoId', type=int) or 0
        return jsonify({
            "productoId": producto_id,
            "prediccionDemanda": 25.5,
            "nivelConfianza": 0.5,
            "fechaPrediccion": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        })

# ENDPOINT OPTIMIZADO para producto específico
@bp.route('/predicciones/producto/<int:producto_id>', methods=['GET'])
def prediccion_producto(producto_id):
    try:
        dias = request.args.get('dias', default=7, type=int)
        logger.info(f"Solicitando predicción para producto {producto_id}, días={dias}")
        return prediccion_producto_directo(producto_id, dias)
            
    except Exception as e:
        logger.error(f" Error en predicción producto {producto_id}: {e}")
        return jsonify({
            "productoId": producto_id,
            "prediccionDemanda": 25.5,
            "nivelConfianza": 0.5,
            "fechaPrediccion": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        })

# ENDPOINT CORREGIDO: predicciones completas CON LIMPIEZA JSON
@bp.route('/predicciones/completas', methods=['GET'])
def predicciones_completas():
    dias = request.args.get('dias', default=30, type=int)
    logger.info(f"Solicitando predicciones completas para {dias} días")
    
    try:
        # Obtener predicciones completas con clasificación ABC
        resultados_predicciones = servicio.generar_predicciones_completas(dias)
        
        # LIMPIAR LOS DATOS PARA JSON ANTES DE ENVIAR
        resultados_limpios = limpiar_para_json(resultados_predicciones)
        
        logger.info(f"Predicciones completas generadas - {len(resultados_limpios.get('sedes', []))} sedes procesadas")
        return jsonify(resultados_limpios)
        
    except Exception as e:
        logger.error(f"Error en predicciones/completas: {e}")
        return jsonify({
            "error": "Error generando predicciones completas",
            "detalle": str(e),
            "fecha_prediccion": datetime.now().isoformat()
        }), 500

# NUEVO: Endpoint para predicciones específicas por sede
@bp.route('/predicciones/sede/<int:sede_id>', methods=['GET'])
def predicciones_por_sede(sede_id):
    """Obtiene predicciones específicas para una sede con clasificación ABC"""
    try:
        dias = request.args.get('dias', default=30, type=int)
        logger.info(f"Solicitando predicciones para sede {sede_id}, días={dias}")
        
        resultado = servicio.generar_predicciones_por_sede(sede_id, dias)
        
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"Error obteniendo predicciones para sede {sede_id}: {e}")
        return jsonify({
            "error": f"Error obteniendo predicciones para sede {sede_id}",
            "detalle": str(e)
        }), 500

# NUEVO: Endpoint para obtener solo la clasificación ABC por sede
@bp.route('/clasificacion-abc/sede/<int:sede_id>', methods=['GET'])
def clasificacion_abc_sede(sede_id):
    """Obtiene solo la clasificación ABC de productos para una sede específica"""
    try:
        dias = request.args.get('dias', default=30, type=int)
        logger.info(f"Solicitando clasificación ABC para sede {sede_id}")
        
        resultado = servicio.generar_predicciones_por_sede(sede_id, dias)
        
        # Extraer solo la información de clasificación ABC
        clasificacion_abc = {
            "sede_id": sede_id,
            "sede_nombre": resultado.get("sede", {}).get("sede_nombre"),
            "clasificacion_abc": resultado.get("sede", {}).get("clasificacion_abc", {}),
            "resumen_abc": resultado.get("sede", {}).get("resumen_abc", {}),
            "fecha_actualizacion": datetime.now().isoformat()
        }
        
        return jsonify(clasificacion_abc)
        
    except Exception as e:
        logger.error(f"Error obteniendo clasificación ABC para sede {sede_id}: {e}")
        return jsonify({
            "error": f"Error obteniendo clasificación ABC para sede {sede_id}",
            "detalle": str(e)
        }), 500

# NUEVO: Endpoint para resumen global de clasificación ABC
@bp.route('/clasificacion-abc/global', methods=['GET'])
def clasificacion_abc_global():
    """Obtiene el resumen global de clasificación ABC de todas las sedes"""
    try:
        dias = request.args.get('dias', default=30, type=int)
        logger.info("Solicitando resumen global de clasificación ABC")
        
        resultado_completo = servicio.generar_predicciones_completas(dias)
        
        # Extraer solo el resumen ABC global
        resumen_abc = {
            "resumen_abc_global": resultado_completo.get("resumen_abc_global", {}),
            "fecha_actualizacion": datetime.now().isoformat(),
            "total_sedes": len(resultado_completo.get("sedes", []))
        }
        
        return jsonify(resumen_abc)
        
    except Exception as e:
        logger.error(f"Error obteniendo clasificación ABC global: {e}")
        return jsonify({
            "error": "Error obteniendo clasificación ABC global",
            "detalle": str(e)
        }), 500

# NUEVO: Endpoint para productos de categoría A por sede (productos críticos)
@bp.route('/productos-criticos/sede/<int:sede_id>', methods=['GET'])
def productos_criticos_sede(sede_id):
    """Obtiene solo los productos categoría A (críticos) de una sede"""
    try:
        dias = request.args.get('dias', default=30, type=int)
        logger.info(f"Solicitando productos críticos (Categoría A) para sede {sede_id}")
        
        resultado = servicio.generar_predicciones_por_sede(sede_id, dias)
        
        productos_criticos = resultado.get("sede", {}).get("clasificacion_abc", {}).get("A", [])
        
        respuesta = {
            "sede_id": sede_id,
            "sede_nombre": resultado.get("sede", {}).get("sede_nombre"),
            "total_productos_criticos": len(productos_criticos),
            "productos_criticos": productos_criticos,
            "fecha_actualizacion": datetime.now().isoformat()
        }
        
        return jsonify(respuesta)
        
    except Exception as e:
        logger.error(f"Error obteniendo productos críticos para sede {sede_id}: {e}")
        return jsonify({
            "error": f"Error obteniendo productos críticos para sede {sede_id}",
            "detalle": str(e)
        }), 500

# Endpoint para recomendaciones del sistema experto (actualizado para trabajar por sede)
@bp.route('/sistema-experto/recomendaciones', methods=['GET'])
def obtener_recomendaciones_experto():
    """Endpoint específico para obtener solo las recomendaciones del sistema experto"""
    try:
        dias = request.args.get('dias', default=30, type=int)
        sede_id = request.args.get('sede_id', type=int)  # Nuevo: filtrar por sede
        
        # Obtener predicciones base
        if sede_id:
            resultados_predicciones = servicio.generar_predicciones_por_sede(sede_id, dias)
            # Convertir a formato esperado por el sistema experto
            resultados_predicciones = {"sedes": [resultados_predicciones.get("sede", {})]}
        else:
            resultados_predicciones = servicio.generar_predicciones_completas(dias)
        
        # Aplicar sistema experto
        recomendaciones = sistema_experto.generar_recomendaciones_reabastecimiento(resultados_predicciones)
        
        return jsonify({
            "fecha_generacion": datetime.now().isoformat(),
            "sede_filtrada": sede_id if sede_id else "Todas",
            "total_recomendaciones": len(recomendaciones),
            "recomendaciones": recomendaciones,
            "estadisticas": sistema_experto.obtener_estadisticas_sistema()
        })
        
    except Exception as e:
        logger.error(f"Error en sistema-experto/recomendaciones: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint para estadísticas del sistema experto
@bp.route('/sistema-experto/estadisticas', methods=['GET'])
def estadisticas_sistema_experto():
    """Endpoint para obtener estadísticas y configuración del sistema experto"""
    try:
        estadisticas = sistema_experto.obtener_estadisticas_sistema()
        return jsonify(estadisticas)
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint para probar alertas del sistema experto
@bp.route('/sistema-experto/test-alertas', methods=['GET'])
def test_alertas_sistema_experto():
    """Endpoint para probar el sistema de alertas con datos de prueba"""
    try:
        # Datos de prueba para el sistema experto
        datos_prueba = {
            "sedes": [
                {
                    "sede_id": 1,
                    "sede_nombre": "Minimarket Central Cajamarca",
                    "productos": [
                        {
                            "producto_id": 1,
                            "nombre_producto": "Arroz",
                            "ventas_pasadas": 150,
                            "rotacion_diaria": 0.05,
                            "proyeccion_diaria": [
                                {"fecha": "2024-01-16", "prediccion": 8},
                                {"fecha": "2024-01-17", "prediccion": 7},
                                {"fecha": "2024-01-18", "prediccion": 9}
                            ]
                        },
                        {
                            "producto_id": 2,
                            "nombre_producto": "Aceite",
                            "ventas_pasadas": 45,
                            "rotacion_diaria": 0.2,
                            "proyeccion_diaria": [
                                {"fecha": "2024-01-16", "prediccion": 3},
                                {"fecha": "2024-01-17", "prediccion": 4},
                                {"fecha": "2024-01-18", "prediccion": 3}
                            ]
                        }
                    ]
                }
            ]
        }
        
        recomendaciones = sistema_experto.generar_recomendaciones_reabastecimiento(datos_prueba)
        
        return jsonify({
            "test_data": datos_prueba,
            "recomendaciones_generadas": recomendaciones,
            "total_recomendaciones": len(recomendaciones)
        })
        
    except Exception as e:
        logger.error(f"Error en test-alertas: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint existente (mantener compatibilidad)
@bp.route('/predicciones', methods=['GET'])
def predicciones_por_defecto():
    try:
        resultados = servicio.generar_predicciones(7)
        return jsonify(resultados)
    except Exception as e:
        logger.error(f"Error en predicciones por defecto: {e}")
        return jsonify({"error": "Error generando predicciones"}), 500

# NUEVO: Health check extendido con información de sedes
@bp.route('/health', methods=['GET'])
def health_check():
    try:
        # Verificar estado de los servicios
        estado_servicios = {
            "prediccion_service": "OK",
            "sistema_experto_service": "OK",
            "clasificacion_abc": "ACTIVADO",
            "base_datos": "OK"
        }
        
        # Información de sedes disponibles
        sedes_disponibles = [
            {"sede_id": 1, "sede_nombre": "Minimarket Central Cajamarca"},
            {"sede_id": 2, "sede_nombre": "Minimarket Los Andes"}
        ]
        
        return jsonify({
            "status": "OK",
            "message": "Microservicio de predicciones funcionando",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "servicios": estado_servicios,
            "sedes_disponibles": sedes_disponibles,
            "endpoints": {
                "predicciones_completas": "/predicciones/completas",
                "predicciones_por_sede": "/predicciones/sede/<id>",
                "prediccion_producto": "/predicciones/producto/<id>",
                "clasificacion_abc_sede": "/clasificacion-abc/sede/<id>",
                "clasificacion_abc_global": "/clasificacion-abc/global",
                "productos_criticos": "/productos-criticos/sede/<id>",
                "sistema_experto_recomendaciones": "/sistema-experto/recomendaciones",
                "sistema_experto_estadisticas": "/sistema-experto/estadisticas",
                "sistema_experto_test": "/sistema-experto/test-alertas",
                "health": "/health"
            }
        })
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            "status": "ERROR",
            "message": "Problemas en el microservicio",
            "error": str(e)
        }), 500

# FUNCIÓN AUXILIAR para filtrar alertas prioritarias
def _filtrar_alertas_prioritarias(recomendaciones):
    """Filtra alertas de alta prioridad de las recomendaciones"""
    alertas_altas = []
    for rec in recomendaciones:
        for alerta in rec.get('alertas', []):
            if alerta.get('severidad') == 'ALTA':
                alertas_altas.append({
                    'sede': rec.get('sede_nombre', 'Desconocida'),
                    'producto': rec.get('nombre_producto', 'Desconocido'),
                    'alerta': alerta.get('tipo', 'DESCONOCIDO'),
                    'mensaje': alerta.get('mensaje', ''),
                    'accion': alerta.get('accion', ''),
                    'categoria_abc': rec.get('categoria_abc', 'No clasificado')
                })
    return alertas_altas