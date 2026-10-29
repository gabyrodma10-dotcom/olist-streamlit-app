from pathlib import Path
from datetime import timedelta
import json
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
st.set_page_config(page_title='Olist: del pedido a la reseña', page_icon='📊', layout='wide')
ROOT = Path(__file__).resolve().parent
API_DEFAULT = 'https://api-olist-riesgo.onrender.com'
UMBRAL_REFERENCIA = 0.262102
st.markdown('\n    <style>\n    .block-container {\n        max-width: 1250px;\n        padding-top: 1.6rem;\n    }\n    h1, h2, h3 {\n        letter-spacing: -.02em;\n    }\n    [data-testid="stMetric"] {\n        border: 1px solid rgba(120,120,120,.22);\n        padding: 16px;\n        border-radius: 12px;\n        background: rgba(120,120,120,.045);\n    }\n    section[data-testid="stSidebar"] {\n        border-right: 1px solid rgba(120,120,120,.15);\n    }\n    </style>\n    ', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cargar_csv(nombre):
    ruta = ROOT / nombre
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, encoding='utf-8-sig', low_memory=False)

def archivo_texto(nombre):
    ruta = ROOT / nombre
    if ruta.exists():
        return ruta.read_text(encoding='utf-8')
    return None

def buscar(df, *nombres):
    if df is None:
        return None
    columnas = {str(c).strip().lower(): c for c in df.columns}
    for nombre in nombres:
        if nombre.lower() in columnas:
            return columnas[nombre.lower()]
    return None

def titulo(nombre, descripcion):
    st.title(nombre)
    st.caption(descripcion)

def csv_descarga(df, nombre, etiqueta):
    st.download_button(etiqueta, data=df.to_csv(index=False).encode('utf-8-sig'), file_name=nombre, mime='text/csv')

def aviso_archivo(nombre):
    st.warning(f'No se encontró **{nombre}** en el repositorio.')

def api_url():
    try:
        return str(st.secrets.get('API_URL', API_DEFAULT)).rstrip('/')
    except Exception:
        return API_DEFAULT
EJEMPLOS = {'Entrega puntual': {'order_id': 'PRUEBA_001', 'order_purchase_timestamp': '2018-05-01T10:00:00', 'order_approved_at': '2018-05-01T11:00:00', 'order_delivered_carrier_date': '2018-05-03T09:00:00', 'order_delivered_customer_date': '2018-05-08T15:00:00', 'order_estimated_delivery_date': '2018-05-12T00:00:00', 'estado_cliente': 'SP', 'n_items': 1, 'n_vendedores': 1, 'precio_total': 120.0, 'flete_total': 15.0, 'peso_total_g': 500.0, 'fotos_promedio': 2.0, 'categoria': 'beleza_saude', 'estado_vendedor': 'SP', 'tipo_pago': 'credit_card', 'cuotas_max': 3}, 'Entrega con retraso': {'order_id': 'PRUEBA_002', 'order_purchase_timestamp': '2018-05-01T10:00:00', 'order_approved_at': '2018-05-01T11:00:00', 'order_delivered_carrier_date': '2018-05-10T09:00:00', 'order_delivered_customer_date': '2018-05-20T18:00:00', 'order_estimated_delivery_date': '2018-05-12T00:00:00', 'estado_cliente': 'SP', 'n_items': 1, 'n_vendedores': 1, 'precio_total': 120.0, 'flete_total': 15.0, 'peso_total_g': 500.0, 'fotos_promedio': 2.0, 'categoria': 'beleza_saude', 'estado_vendedor': 'SP', 'tipo_pago': 'credit_card', 'cuotas_max': 3}}

def inicio():
    st.caption('AIMLF1 · PROYECTO FINAL DE MACHINE LEARNING')
    izquierda, derecha = st.columns([1.15, 1], gap='large', vertical_alignment='center')
    with izquierda:
        st.title('Olist: del pedido a la reseña')
        st.write('Este proyecto integra modelos de aprendizaje automático y análisis de datos para identificar patrones de riesgo, comprender la satisfacción de los clientes y detectar oportunidades de mejora en la operación de los vendedores.')
        st.caption('Predicción de reseñas · Análisis de comentarios · Segmentación de vendedores')
    with derecha:
        imagen = ROOT / 'olist_resenas_portada.png'
        if imagen.exists():
            st.image(str(imagen), use_container_width=True)
        else:
            st.info('Agrega olist_resenas_portada.png a la raíz del repositorio.')
    vendedores = cargar_csv('segmentacion_vendedores.csv')
    comentarios_df = cargar_csv('sentimiento_comentarios.csv')
    a, b, c = st.columns(3)
    a.metric('Vendedores analizados', f'{len(vendedores):,}' if vendedores is not None else 'No disponible')
    b.metric('Comentarios analizados', f'{len(comentarios_df):,}' if comentarios_df is not None else 'No disponible')
    c.metric('Áreas de análisis', '3')
    st.subheader('¿Qué encontrarás en esta aplicación?')
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('#### 📦 Predicción de riesgo')
            st.write('Simula un pedido y consulta la probabilidad estimada de recibir una reseña de 1 o 2 estrellas mediante XGBoost y FastAPI.')
        with st.container(border=True):
            st.markdown('#### 👥 Segmentación de vendedores')
            st.write('Explora los grupos identificados con K-Means y sus diferencias en pedidos, ventas, categorías, retrasos y costos de envío.')
    with col2:
        with st.container(border=True):
            st.markdown('#### 💬 Análisis de comentarios')
            st.write('Compara el sentimiento expresado en las reseñas con las estrellas otorgadas por los clientes.')
        with st.container(border=True):
            st.markdown('#### 📈 Validación y comparación de modelos')
            st.write('Revisa el desempeño de XGBoost, regresión logística y las referencias de comparación.')
    with st.container(border=True):
        st.markdown('#### 🧠 Plus · Keras vs. XGBoost (+5 puntos)')
        st.write('Explora la comparación adicional de una red neuronal Keras frente a XGBoost, con early stopping y curva de pérdida.')
    st.subheader('¿Cómo usar la aplicación?')
    st.markdown('**1.** Selecciona una sección en el menú lateral.  \n**2.** Explora los resultados o prueba un pedido en el simulador.  \n**3.** Consulta la validación y la ficha del modelo para profundizar en la metodología.')

def riesgo():
    titulo('⚠️ Predicción de riesgo', 'Consulta el modelo XGBoost desplegado en FastAPI y prioriza pedidos según su riesgo de reseña negativa.')
    st.write('El modelo estima la probabilidad de que un pedido reciba una reseña de **1 o 2 estrellas**, a partir de sus características y datos de entrega. Para la priorización se utiliza el **10% de pedidos con mayor riesgo** según un umbral establecido con datos de entrenamiento.')
    st.info('**Demostración posterior a la entrega:** se usan fechas reales de entrega. No es una predicción disponible antes de entregar el pedido. No se utiliza texto, longitud ni fecha de la reseña.')
    caso = st.radio('1. Selecciona un escenario de ejemplo', ['📦 Entrega puntual', '🚚 Entrega con retraso'], horizontal=True, help='Pedidos ficticios para probar el funcionamiento de la API.')
    clave = 'Entrega puntual' if caso.startswith('📦') else 'Entrega con retraso'
    datos = EJEMPLOS[clave].copy()
    pref = 'riesgo_' + clave.replace(' ', '_')
    campos_fecha = [('order_purchase_timestamp', 'Fecha de compra'), ('order_approved_at', 'Fecha de aprobación'), ('order_delivered_carrier_date', 'Entrega al transportista'), ('order_estimated_delivery_date', 'Entrega prometida'), ('order_delivered_customer_date', 'Entrega real')]
    for campo, _ in campos_fecha:
        key = f'{pref}_{campo}'
        if key not in st.session_state:
            st.session_state[key] = pd.to_datetime(datos[campo]).date()
    key_dias = f'{pref}_dias'
    key_real = f'{pref}_order_delivered_customer_date'
    key_prometida = f'{pref}_order_estimated_delivery_date'
    if key_dias not in st.session_state:
        st.session_state[key_dias] = (st.session_state[key_real] - st.session_state[key_prometida]).days

    def cambio_barra():
        st.session_state[key_real] = st.session_state[key_prometida] + timedelta(days=st.session_state[key_dias])

    def cambio_fecha_real():
        diferencia = (st.session_state[key_real] - st.session_state[key_prometida]).days
        st.session_state[key_dias] = max(-60, min(90, diferencia))

    def cambio_fecha_prometida():
        cambio_fecha_real()
    st.subheader('2. Revisa o cambia el pedido')
    st.caption('Ejemplos hipotéticos: cambiar fechas no incorpora pedidos ocultos al entrenamiento ni constituye una evaluación de esos pedidos.')
    st.slider('¿Cuántos días antes o después de la fecha prometida llegó el pedido?', min_value=-60, max_value=90, step=1, key=key_dias, on_change=cambio_barra, help='-4 significa cuatro días antes; +8, ocho días tarde.')
    prometida = st.session_state[key_prometida]
    real = st.session_state[key_real]
    dias = (real - prometida).days
    a, b, c = st.columns(3)
    a.metric('Fecha prometida', prometida.strftime('%d/%m/%Y'))
    b.metric('Entrega real', real.strftime('%d/%m/%Y'))
    c.metric('Días respecto al plazo', f'{dias:+d}')
    if dias > 0:
        st.warning(f'El pedido llega **{dias} días tarde**.')
    elif dias < 0:
        st.success(f'El pedido llega **{abs(dias)} días antes**.')
    else:
        st.success('El pedido llega en la fecha prometida.')
    with st.expander('⚙️ Ajustes avanzados (opcional)'):
        st.caption('Puedes cambiar las fechas una por una o usar la barra superior. Al modificar la entrega real, la barra también se actualiza.')
        izquierda, derecha = st.columns(2)
        with izquierda:
            for campo, etiqueta in [('order_id', 'ID del pedido'), ('estado_cliente', 'Estado del cliente'), ('estado_vendedor', 'Estado del vendedor'), ('categoria', 'Categoría'), ('tipo_pago', 'Tipo de pago')]:
                datos[campo] = st.text_input(etiqueta, value=str(datos[campo]), key=f'{pref}_{campo}')
            for campo, etiqueta in [('n_items', 'Número de artículos'), ('n_vendedores', 'Número de vendedores'), ('cuotas_max', 'Cuotas máximas')]:
                datos[campo] = int(st.number_input(etiqueta, min_value=0, value=int(datos[campo]), step=1, key=f'{pref}_{campo}'))
        with derecha:
            st.markdown('**Fechas del pedido (editables)**')
            for campo, etiqueta in campos_fecha:
                kwargs = {}
                if campo == 'order_delivered_customer_date':
                    kwargs['on_change'] = cambio_fecha_real
                elif campo == 'order_estimated_delivery_date':
                    kwargs['on_change'] = cambio_fecha_prometida
                st.date_input(etiqueta, min_value=pd.Timestamp('2016-01-01').date(), max_value=pd.Timestamp('2028-12-31').date(), key=f'{pref}_{campo}', **kwargs)
            for campo, etiqueta in [('precio_total', 'Precio total'), ('flete_total', 'Costo de envío'), ('peso_total_g', 'Peso en gramos'), ('fotos_promedio', 'Fotos promedio')]:
                datos[campo] = float(st.number_input(etiqueta, min_value=0.0, value=float(datos[campo]), key=f'{pref}_{campo}'))
    for campo, _ in campos_fecha:
        original = pd.to_datetime(EJEMPLOS[clave][campo])
        fecha = st.session_state[f'{pref}_{campo}']
        datos[campo] = pd.Timestamp.combine(fecha, original.time()).strftime('%Y-%m-%dT%H:%M:%S')
    compra = pd.to_datetime(datos['order_purchase_timestamp'])
    aprobacion = pd.to_datetime(datos['order_approved_at'])
    transportista = pd.to_datetime(datos['order_delivered_carrier_date'])
    entrega = pd.to_datetime(datos['order_delivered_customer_date'])
    errores = []
    if aprobacion < compra: errores.append('La aprobación no puede ser anterior a la compra.')
    if transportista < aprobacion: errores.append('La entrega al transportista no puede ser anterior a la aprobación.')
    if entrega < transportista: errores.append('La entrega real no puede ser anterior a la entrega al transportista.')
    for error in errores: st.error(error)
    enviar = st.button('🔎 Calcular riesgo del pedido', type='primary', use_container_width=True, disabled=bool(errores), key=f'{pref}_enviar')
    with st.expander('🧪 Prueba técnica de la API (JSON editable)'):
        st.caption('Verifica el contrato de entrada y prueba el endpoint /riesgo.')
        texto = st.text_area('JSON del pedido', value=json.dumps(datos, indent=2, ensure_ascii=False), height=280, key=f'{pref}_json')
        enviar_json = st.button('Enviar JSON técnico', key=f'{pref}_json_enviar')
    if enviar_json:
        try:
            datos = json.loads(texto)
            if not isinstance(datos, dict):
                st.error('El JSON debe ser un objeto con los campos del pedido.')
                return
        except json.JSONDecodeError as exc:
            st.error(f'JSON inválido: {exc}')
            return
    elif not enviar:
        return
    try:
        compra = pd.to_datetime(datos['order_purchase_timestamp'])
        aprobacion = pd.to_datetime(datos['order_approved_at'])
        transportista = pd.to_datetime(datos['order_delivered_carrier_date'])
        entrega = pd.to_datetime(datos['order_delivered_customer_date'])
        estimada = pd.to_datetime(datos['order_estimated_delivery_date'])
    except (ValueError, KeyError, TypeError):
        st.error('Revisa las fechas: falta un campo o alguna fecha no es válida.')
        return
    if not compra <= aprobacion <= transportista <= entrega:
        st.error('Las fechas deben seguir este orden: compra, aprobación, transportista y entrega real.')
        return
    if estimada < compra:
        st.error('La fecha prometida no puede ser anterior a la compra.')
        return
    with st.spinner('Consultando el modelo; Render puede tardar en activarse...'):
        try:
            respuesta = requests.post(api_url() + '/riesgo', json=datos, timeout=120)
        except requests.exceptions.Timeout:
            st.error('La API tardó demasiado. Vuelve a intentarlo en unos segundos.')
            return
        except requests.exceptions.RequestException as exc:
            st.error('No se pudo conectar con la API.')
            st.caption(str(exc))
            return
    if respuesta.status_code == 422:
        st.warning('La API rechazó los datos (HTTP 422). Revisa los campos y las fechas.')
        try:
            st.json(respuesta.json())
        except ValueError:
            st.code(respuesta.text)
        return
    if not respuesta.ok:
        st.error(f'La API devolvió HTTP {respuesta.status_code}.')
        st.code(respuesta.text[:1000])
        return
    try:
        resultado = respuesta.json()
        probabilidad = float(resultado['probabilidad'])
        if not 0 <= probabilidad <= 1:
            raise ValueError('Probabilidad fuera de rango')
    except (ValueError, TypeError, KeyError):
        st.error('La API no devolvió una probabilidad válida.')
        try:
            st.json(respuesta.json())
        except ValueError:
            st.code(respuesta.text)
        return
    alto = resultado.get('riesgo_alto')
    umbral_api = resultado.get('umbral_riesgo', resultado.get('umbral'))
    try:
        umbral = float(umbral_api) if umbral_api is not None else UMBRAL_REFERENCIA
        if not 0 <= umbral <= 1:
            raise ValueError('Umbral fuera de rango')
    except (ValueError, TypeError):
        umbral = UMBRAL_REFERENCIA
    st.subheader('3. Resultado de la predicción')
    izquierda, derecha = st.columns(2)
    izquierda.metric('Probabilidad estimada de 1–2 estrellas', f'{probabilidad:.2%}')
    derecha.metric('Umbral de riesgo alto (Top 10%)', f'{umbral:.2%}')
    if alto is True:
        st.error('**RIESGO ALTO · Priorizar seguimiento.** El pedido supera el criterio de la API.')
    elif alto is False:
        st.success('**RIESGO NO ALTO.** El pedido no fue priorizado por la API.')
    else:
        st.warning('La API no devolvió una clasificación de riesgo alto/no alto.')
    if umbral_api is None:
        st.caption('El umbral mostrado (26.21%) proviene del cálculo final del notebook; la clasificación se toma directamente de la API. Comprueba que el umbral configurado en main.py sea el mismo.')
    else:
        st.caption('El umbral mostrado y la clasificación proceden de la API desplegada.')
    st.markdown('**¿Cómo interpretar el resultado?**')
    st.write('La probabilidad estima el riesgo individual de una reseña negativa. El umbral sirve para decidir qué pedidos requieren atención prioritaria. No significa que el 26.21% de los pedidos recibirá una mala reseña.')
    st.caption('El 10% se define al ordenar las probabilidades del conjunto de referencia; no implica que exactamente el 10% de cada lote de pruebas quede clasificado como alto. Los ejemplos son simulados y la evaluación del docente utiliza pedidos ocultos.')
    with st.expander('Ver respuesta técnica completa de FastAPI'):
        st.json(resultado)

def comentarios():
    titulo('💬 Análisis de comentarios', 'Sentimiento del texto frente a las estrellas')
    df = cargar_csv('sentimiento_comentarios.csv')
    if df is None:
        aviso_archivo('sentimiento_comentarios.csv')
        return
    requeridas = ['review_score', 'review_comment_message', 'sentimiento_modelo']
    if any(c not in df.columns for c in requeridas):
        st.error('El CSV no contiene las columnas requeridas: ' + ', '.join(requeridas))
        return
    datos = df.copy()
    datos['review_score'] = pd.to_numeric(datos['review_score'], errors='coerce')
    datos['sentimiento_modelo'] = datos['sentimiento_modelo'].astype(str).str.lower().str.strip()
    datos['sentimiento_estrellas'] = datos['review_score'].map(
        lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive') if pd.notna(x) else None
    )
    datos['coincide_calculado'] = datos['sentimiento_modelo'] == datos['sentimiento_estrellas']
    validos = datos['review_score'].between(1, 5) & datos['sentimiento_modelo'].isin(['negative','neutral','positive'])
    datos = datos.loc[validos].copy()
    if datos.empty:
        st.error('No hay filas válidas para comparar estrellas y sentimiento.')
        return
    aciertos = int(datos['coincide_calculado'].sum())
    diferencias = len(datos) - aciertos
    a,b,c = st.columns(3)
    a.metric('Comentarios analizados', f'{len(datos):,}')
    b.metric('Coincidencia texto-estrellas', f'{aciertos/len(datos):.2%}')
    c.metric('Desacuerdos', f'{diferencias:,}')
    st.caption('Muestra reproducible del notebook: 1,000 comentarios (random_state=42). Modelo preentrenado: cardiffnlp/twitter-xlm-roberta-base-sentiment. Las etiquetas ya están calculadas en Colab; la app no vuelve a ejecutar el modelo.')
    st.subheader('Distribución de sentimientos')
    conteo = datos['sentimiento_modelo'].value_counts().rename_axis('Sentimiento').reset_index(name='Comentarios')
    st.plotly_chart(px.bar(conteo, x='Sentimiento', y='Comentarios', color='Sentimiento', text='Comentarios'), use_container_width=True)
    st.subheader('Sentimiento frente a estrellas')
    tabla = pd.crosstab(datos['review_score'].astype(int), datos['sentimiento_modelo']).reindex(index=[1,2,3,4,5], columns=['negative','neutral','positive'], fill_value=0)
    tabla.index.name = 'Estrellas'
    st.dataframe(tabla, use_container_width=True)
    st.plotly_chart(px.imshow(tabla, text_auto=True, aspect='auto', labels={'x':'Sentimiento del texto','y':'Estrellas','color':'Comentarios'}), use_container_width=True)
    st.caption('Criterio: 1–2 estrellas = negativo; 3 = neutro; 4–5 = positivo. La tabla presenta cantidades, no porcentajes.')
    st.subheader('¿Dónde aparecen las diferencias?')
    positivos = datos[datos['review_score'].isin([4,5])]
    neutros_altos = int((positivos['sentimiento_modelo']=='neutral').sum())
    negativos_altos = int((positivos['sentimiento_modelo']=='negative').sum())
    st.write(f'En **{len(datos):,}** comentarios, **{aciertos:,}** coinciden y **{diferencias:,}** no coinciden. Entre las reseñas de 4 y 5 estrellas, **{neutros_altos}** textos fueron clasificados como neutrales y **{negativos_altos}** como negativos. Una diferencia no demuestra por sí sola un error: las estrellas valoran la experiencia general y el texto puede referirse a un aspecto concreto.')
    st.subheader('Cinco desacuerdos reales e interpretados')
    # Casos analizados previamente en el notebook; solo se muestran si el CSV confirma el caso.
    casos = [
        ('Blz', 4, 'neutral', 'Posible limitación del modelo: expresión demasiado breve para detectar claramente un tono positivo.'),
        ('Achei só muito demorado o prazo para a entrega.', 4, 'negative', 'Diferencia entre valoración general y queja concreta: el texto critica el plazo de entrega pese a las cuatro estrellas.'),
        ('Produto não entregue', 1, 'neutral', 'Posible error del modelo: la frase describe una entrega fallida, claramente desfavorable, pero fue clasificada como neutra.'),
        ('Tudo ok. Exceto que a caixa de dentro veio violada.', 4, 'negative', 'Experiencia mixta: el cliente evalúa favorablemente el conjunto, pero informa un problema de embalaje.'),
        ('fornecedor correto e produto excelente', 3, 'positive', 'El texto elogia al proveedor y al producto, mientras que las tres estrellas sugieren que pudo haber otros factores no mencionados.'),
    ]
    encontrados = []
    texto_normalizado = datos['review_comment_message'].fillna('').astype(str).str.strip().str.casefold()
    for fragmento, estrellas, etiqueta, interpretacion in casos:
        mascara = texto_normalizado.str.contains(fragmento.casefold(), regex=False) & (datos['review_score']==estrellas) & (datos['sentimiento_modelo']==etiqueta)
        coincidencias = datos.loc[mascara]
        if not coincidencias.empty:
            fila = coincidencias.iloc[0]
            encontrados.append((fila, interpretacion))
    for i, (fila, interpretacion) in enumerate(encontrados, 1):
        with st.container(border=True):
            st.markdown(f'**Caso {i} · {int(fila["review_score"])} estrellas · Modelo: {fila["sentimiento_modelo"]}**')
            st.write('Comentario original: ' + str(fila['review_comment_message']))
            st.write('**Interpretación:** ' + interpretacion)
    if len(encontrados)<5:
        st.warning(f'Se verificaron {len(encontrados)} de los 5 casos documentados en el notebook. Comprueba que el CSV publicado corresponde a esa ejecución; no se muestran interpretaciones para comentarios no verificados.')
    st.subheader('¿Qué aporta el texto frente a las estrellas?')
    st.write('El texto permite identificar **causas específicas** de satisfacción o insatisfacción —por ejemplo, retrasos, entregas fallidas, embalajes dañados o recomendaciones— que una puntuación numérica no explica. Un cliente puede asignar cuatro estrellas y aun así describir un problema logístico. Por eso, para Olist conviene analizar **estrellas y comentarios en conjunto**: las estrellas resumen la valoración y el texto ayuda a orientar acciones de mejora.')
    st.write('**Limitaciones:** los textos cortos, las opiniones mixtas y el contexto lingüístico pueden generar desacuerdos. El 70.50% de coincidencia es una comparación con las estrellas, no una medida de exactitud validada frente a etiquetas humanas.')
    with st.expander('Explorar los comentarios y las etiquetas'):
        mostrar = [c for c in ['review_id','order_id','review_score','review_comment_message','sentimiento_estrellas','sentimiento_modelo','confianza_modelo','coincide_calculado'] if c in datos.columns]
        st.dataframe(datos[mostrar], use_container_width=True, hide_index=True)
        csv_descarga(datos, 'sentimiento_comentarios_analisis.csv', 'Descargar análisis de comentarios')

def segmentos():
    titulo('👥 Segmentación de vendedores', 'Agrupación K-Means · Parte 3B')
    df = cargar_csv('segmentacion_vendedores.csv')
    if df is None:
        aviso_archivo('segmentacion_vendedores.csv')
        return
    segmento = buscar(df, 'segmento', 'cluster', 'segment')
    if segmento is None:
        st.error('El CSV no contiene la columna segmento.')
        return
    identificador = buscar(df, 'seller_id')
    a, b = st.columns(2)
    a.metric('Segmentos identificados', str(df[segmento].nunique()))
    b.metric('Vendedores analizados', f'{df[identificador].nunique():,}' if identificador else f'{len(df):,}')
    tamanos = df.groupby(segmento, dropna=False).size().rename('Vendedores').reset_index()
    st.subheader('Distribución de vendedores')
    fig = px.bar(tamanos, x=segmento, y='Vendedores', color=segmento, text='Vendedores')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Perfiles cuantitativos por segmento')
    excluir = {segmento, identificador}
    numericas = []
    trabajo = df.copy()
    for columna in df.columns:
        if columna in excluir:
            continue
        convertido = pd.to_numeric(df[columna], errors='coerce')
        minimo = max(5, int(len(df) * 0.5))
        if convertido.notna().sum() >= minimo:
            trabajo[columna] = convertido
            numericas.append(columna)
    if numericas:
        elegidas = st.multiselect('Indicadores para comparar', numericas, default=numericas[:min(5, len(numericas))])
        if elegidas:
            perfiles = trabajo.groupby(segmento)[elegidas].mean().round(3)
            st.dataframe(perfiles, use_container_width=True)
            largo = perfiles.reset_index().melt(id_vars=segmento, var_name='Indicador', value_name='Promedio')
            fig = px.bar(largo, x=segmento, y='Promedio', color='Indicador', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Los indicadores tienen distintas unidades. Conviene comparar cada indicador entre segmentos.')
    else:
        st.warning('No se encontraron indicadores cuantitativos suficientes.')
    st.subheader('Nombres y acciones por segmento')
    st.caption('Los perfiles se describen a partir de los indicadores de pedidos, ventas, diversidad de categorías, retrasos y costos de envío. Los nombres y acciones representan propuestas de negocio basadas en las características observadas de cada grupo.')
    campos = ['total_pedidos', 'ventas_totales', 'n_categorias', 'tasa_retraso', 'flete_relativo']
    faltan = [campo for campo in campos if campo not in trabajo.columns]
    if faltan:
        st.warning('No se pueden generar los perfiles: faltan los indicadores ' + ', '.join(faltan))
    else:
        base = trabajo.groupby(segmento)[campos].mean()
        medianas = trabajo[campos].median()
        registros = []
        for grupo, fila in base.iterrows():
            volumen_alto = fila['total_pedidos'] > medianas['total_pedidos']
            ventas_altas = fila['ventas_totales'] > medianas['ventas_totales']
            retrasos_altos = fila['tasa_retraso'] > medianas['tasa_retraso']
            flete_alto = fila['flete_relativo'] > medianas['flete_relativo']
            if retrasos_altos and volumen_alto:
                nombre = 'Alto volumen con retos de entrega'
                accion = 'Priorizar revisión logística, tiempos de despacho y seguimiento de pedidos.'
            elif retrasos_altos:
                nombre = 'Vendedores con oportunidad logística'
                accion = 'Investigar las causas de retrasos y definir acciones de mejora.'
            elif volumen_alto and ventas_altas:
                nombre = 'Vendedores consolidados'
                accion = 'Mantener buenas prácticas y fortalecer oportunidades de crecimiento.'
            elif flete_alto:
                nombre = 'Vendedores con flete relativo alto'
                accion = 'Revisar costos y alternativas de envío.'
            else:
                nombre = 'Vendedores de operación moderada'
                accion = 'Monitorear resultados e identificar oportunidades de desarrollo.'
            perfil = f"Pedidos promedio: {fila['total_pedidos']:.1f}; ventas promedio: {fila['ventas_totales']:.2f}; categorías promedio: {fila['n_categorias']:.1f}; tasa de retraso promedio: {fila['tasa_retraso']:.3f}; flete relativo promedio: {fila['flete_relativo']:.3f}."
            registros.append({'Segmento': grupo, 'Nombre propuesto': nombre, 'Perfil': perfil, 'Acción recomendada': accion})
        tabla_perfiles = pd.DataFrame(registros)
        st.dataframe(tabla_perfiles, use_container_width=True, hide_index=True)
    st.subheader('Explorador de vendedores')
    opciones = sorted(df[segmento].dropna().unique(), key=str)
    elegido = st.selectbox('Filtrar segmento', ['Todos'] + list(opciones))
    if elegido == 'Todos':
        filtrado = df
    else:
        filtrado = df[df[segmento] == elegido]
    st.dataframe(filtrado, use_container_width=True)
    csv_descarga(filtrado, 'segmentacion_filtrada.csv', 'Descargar vendedores filtrados')
    st.caption('La selección de K, el preprocesamiento y la interpretación del PCA se documentan en el notebook.')

def validacion():
    titulo('📈 Validación y resultados', 'Comparación de modelos predictivos')
    st.write('La evaluación temporal permite comparar el desempeño de los modelos sobre un periodo de validación posterior.')
    a, b, c = st.columns(3)
    a.metric('Pedidos de entrenamiento', '57,065')
    b.metric('Pedidos de validación', '15,461')
    c.metric('Precisión Top 10% · XGBoost', '49.48%')
    resultados = pd.DataFrame({'Método': ['Selección aleatoria', 'Regla del retraso', 'Regresión logística', 'XGBoost'], 'Precisión Top 10%': [11.89, 38.55, 45.99, 49.48]})
    st.subheader('Comparación de desempeño')
    fig = px.bar(resultados, x='Método', y='Precisión Top 10%', text_auto='.2f', title='Precisión entre el 10% de pedidos priorizados')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(resultados, use_container_width=True, hide_index=True)
    st.success('XGBoost obtuvo la mayor precisión Top 10% entre los métodos comparados.')
    st.write('En esta validación, aproximadamente 5 de cada 10 pedidos priorizados por XGBoost recibieron una reseña negativa.')
    st.caption('Los valores mostrados corresponden a resultados previamente reportados en el notebook del proyecto.')
    with st.expander('Consideraciones de validación'):
        st.write('La evaluación debe mantener la separación temporal, comprobar la ausencia de fuga de información y utilizar el mismo criterio de comparación para los modelos.')
        st.write('Los resultados de validación no equivalen a una evaluación sobre datos ocultos del docente.')

def plus():
    titulo('🧠 Plus · Keras vs. XGBoost', 'Comparación adicional de la rúbrica (+5 puntos)')
    st.write('Se compara una red neuronal densa con XGBoost, utilizando las mismas variables procesadas y el mismo conjunto de validación temporal.')
    a, b, c = st.columns(3)
    a.metric('Pedidos de validación', '15,461')
    b.metric('Características procesadas', '140')
    c.metric('Mejor época de Keras', '5 de 10')
    st.subheader('Precisión Top 10%')
    comparacion = pd.DataFrame({'Modelo': ['Red neuronal Keras', 'XGBoost'], 'Precisión Top 10% (%)': [48.67, 48.93]})
    fig = px.bar(comparacion, x='Modelo', y='Precisión Top 10% (%)', color='Modelo', text_auto='.2f', range_y=[0, 60])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comparacion, use_container_width=True, hide_index=True)
    st.success('En esta comparación, XGBoost superó a Keras por 0.26 puntos porcentuales.')
    st.caption('La comparación adicional utiliza 48.93% para XGBoost, correspondiente a la ejecución comparativa con Keras. La validación principal reporta 49.48%.')
    st.subheader('Curva de pérdida por época')
    perdidas = pd.DataFrame({'Época': list(range(1, 11)), 'Entrenamiento': [0.3591, 0.3265, 0.3215, 0.3177, 0.3147, 0.3125, 0.3107, 0.309, 0.3074, 0.3058], 'Validación': [0.3109, 0.3068, 0.3051, 0.3044, 0.3039, 0.304, 0.3043, 0.3047, 0.3055, 0.3061]})
    grafico = perdidas.melt(id_vars='Época', var_name='Conjunto', value_name='Pérdida')
    fig = px.line(grafico, x='Época', y='Pérdida', color='Conjunto', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.info('La menor pérdida de validación reportada fue 0.3039 en la época 5. El entrenamiento utilizó early stopping.')
    st.subheader('Arquitectura de la red neuronal')
    st.write('Entrada de 140 características, capa densa de 64 neuronas ReLU, capa densa de 32 neuronas ReLU y salida sigmoide.')
    st.write('Optimizador Adam, función de pérdida binary crossentropy, tamaño de lote 256 y paciencia de 5 épocas.')
    st.subheader('Conclusión')
    st.write('Los resultados fueron cercanos. XGBoost presentó una ventaja pequeña en la precisión Top 10% de esta comparación, por lo que se mantuvo como modelo de la API.')
    st.caption('Los valores de la curva corresponden al historial de entrenamiento previamente reportado, redondeado a cuatro decimales.')

def ficha():
    titulo('📄 Ficha del modelo', 'Características, resultados y limitaciones')
    texto = archivo_texto('ficha_modelo.md')
    if texto:
        st.markdown(texto)
        st.download_button('Descargar ficha', data=texto, file_name='ficha_modelo.md', mime='text/markdown')
        return
    st.subheader('1. Propósito y uso previsto')
    st.write('El modelo estima la probabilidad de que un pedido reciba una reseña de 1 o 2 estrellas y permite priorizar casos para su revisión.')
    st.subheader('2. Datos utilizados')
    st.write('El proyecto utiliza información histórica del conjunto de datos Olist. La validación principal reportada incluye 15,461 pedidos.')
    st.subheader('3. Desempeño del modelo')
    st.write('Precisión Top 10% reportada: XGBoost 49.48%, regresión logística 45.99%, regla del retraso 38.55% y selección aleatoria 11.89%.')
    st.subheader('4. Limitaciones')
    st.write('El rendimiento puede cambiar ante nuevas condiciones de entrega, categorías poco representadas o diferencias entre los datos históricos y los pedidos futuros.')
    st.write('El modelo utiliza la fecha real de entrega. Por ello, no puede presentarse como una predicción disponible antes de entregar el pedido.')
    st.subheader('5. Posibles errores e impacto')
    st.write('Un falso positivo puede ocasionar un seguimiento innecesario, mientras que un falso negativo puede dejar sin atención prioritaria a un cliente insatisfecho.')
    st.subheader('6. Recomendaciones de uso')
    st.write('Utilizar el modelo como apoyo analítico, revisar periódicamente su desempeño y mantener supervisión humana en las decisiones operativas.')
    st.caption('Esta ficha integrada es una síntesis para la aplicación. La versión final de entrega debe contrastarse con los resultados y la documentación del notebook.')
paginas = [st.Page(inicio, title='Inicio', icon='🏠'), st.Page(riesgo, title='Predicción de riesgo', icon='⚠️'), st.Page(comentarios, title='Análisis de comentarios', icon='💬'), st.Page(segmentos, title='Segmentación 3B', icon='👥'), st.Page(validacion, title='Validación y resultados', icon='📈'), st.Page(plus, title='Plus · Keras vs. XGBoost', icon='🧠'), st.Page(ficha, title='Ficha del modelo', icon='📄')]
st.navigation(paginas).run()
st.divider()
st.caption('Proyecto final Olist · Predicción de reseñas y análisis de vendedores')
