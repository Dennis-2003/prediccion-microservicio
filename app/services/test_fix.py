from app.services.prediccion_service import PrediccionService
import pandas as pd
from app.utils.data_loader import cargar_ventas, cargar_detalles, cargar_productos

# Crear una versión corregida
servicio = PrediccionService()

def fixed_method(dias=30):
    print("=== USANDO VERSIÓN CORREGIDA ===")
    
    # Cargar datos
    ventas = cargar_ventas()
    detalles = cargar_detalles()
    productos = cargar_productos()

    ventas_df = pd.DataFrame(ventas)
    detalles_df = pd.DataFrame(detalles)
    productos_df = pd.DataFrame(productos)

    if ventas_df.empty or detalles_df.empty or productos_df.empty:
        return {"error": True, "mensaje": "Datos insuficientes", "sedes": []}

    # Procesamiento normal
    ventas_df.rename(columns={ventas_df.columns[0]: 'venta_id'}, inplace=True)
    detalles_df.rename(columns={detalles_df.columns[0]: 'detalle_id'}, inplace=True)
    
    # Merge corregido
    detalles_df = detalles_df.merge(productos_df, on='producto_id', how='left')
    data = detalles_df.merge(ventas_df, on='venta_id', how='inner')
    
    # CORRECCIÓN CLAVE: Unificar sede_id
    if 'sede_id_y' in data.columns:
        data['sede_id'] = data['sede_id_y']  # Usar sede de ventas (que tiene sedes 1 y 2)
        print(f" Usando sede_id de ventas. Sedes encontradas: {data['sede_id'].unique()}")
    else:
        data['sede_id'] = 1
    
    data = data.merge(servicio.sedes_df, on='sede_id', how='left')

    print(f" Datos procesados: {len(data)} registros")

    # Generar predicciones por sede
    sedes_predicciones = {}
    for (sede_id, sede_nombre), grupo_sede in data.groupby(["sede_id", "sede_nombre"]):
        print(f"Procesando sede: {sede_nombre} (ID: {sede_id}) - {len(grupo_sede)} registros")
        sede_info = servicio._procesar_sede(sede_id, sede_nombre, grupo_sede, dias)
        sedes_predicciones[sede_id] = sede_info

    resultado_predicciones = {
        "fecha_prediccion": pd.Timestamp.now().strftime('%Y-%m-%d'),
        "horizonte_dias": dias,
        "sedes": list(sedes_predicciones.values())
    }
    
    print(f"Procesamiento completado: {len(sedes_predicciones)} sedes")
    return resultado_predicciones

# Probar la versión corregida
print("=== PROBANDO VERSIÓN CORREGIDA ===")
resultado = fixed_method(7)
print(f" RESULTADO: {len(resultado.get('sedes', []))} sedes procesadas")

if resultado.get('sedes'):
    for sede in resultado['sedes']:
        print(f" {sede['sede_nombre']}: {len(sede['productos'])} productos, Ganancia: {sede['ganancia_total']}")
else:
    print("No se procesaron sedes")