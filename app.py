from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

st.set_page_config(page_title='Olist | Inteligencia de reseñas',page_icon='📊',layout='wide')
ROOT=Path(__file__).resolve().parent
API_DEFAULT='https://api-olist-riesgo.onrender.com'

st.markdown('''<style>.block-container{max-width:1250px;padding-top:1.6rem}h1,h2,h3{letter-spacing:-.02em}[data-testid="stMetric"]{border:1px solid rgba(120,120,120,.22);padding:16px;border-radius:12px;background:rgba(120,120,120,.045)}section[data-testid="stSidebar"]{border-right:1px solid rgba(120,120,120,.15)}</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cargar_csv(nombre):
    ruta=ROOT/nombre
    if not ruta.exists(): return None
    return pd.read_csv(ruta,encoding='utf-8-sig',low_memory=False)

def archivo_texto(nombre):
    ruta=ROOT/nombre
    return ruta.read_text(encoding='utf-8') if ruta.exists() else None

def buscar(df,*nombres):
    if df is None:return None
    cols={str(c).strip().lower():c for c in df.columns}
    for nombre in nombres:
        if nombre.lower() in cols:return cols[nombre.lower()]
    return None

def titulo(nombre,descripcion):
    st.title(nombre)
    st.caption(descripcion)

def csv_descarga(df,nombre,etiqueta):
    st.download_button(etiqueta,df.to_csv(index=False).encode('utf-8-sig'),nombre,mime='text/csv')

def aviso_archivo(nombre):
    st.warning(f'Falta **{nombre}** en el repositorio de GitHub. Agrégalo para habilitar esta sección.')

def api_url():
    try:return str(st.secrets.get('API_URL',API_DEFAULT)).rstrip('/')
    except Exception:return API_DEFAULT

EJEMPLOS={
    'Riesgo bajo (entrega puntual)':{
        'order_id':'PRUEBA_001','order_purchase_timestamp':'2018-05-01T10:00:00',
        'order_approved_at':'2018-05-01T11:00:00','order_delivered_carrier_date':'2018-05-03T09:00:00',
        'order_delivered_customer_date':'2018-05-08T15:00:00','order_estimated_delivery_date':'2018-05-12T00:00:00',
        'estado_cliente':'SP','n_items':1,'n_vendedores':1,'precio_total':120.0,'flete_total':15.0,
        'peso_total_g':500.0,'fotos_promedio':2.0,'categoria':'beleza_saude',
        'estado_vendedor':'SP','tipo_pago':'credit_card','cuotas_max':3},
    'Riesgo alto (entrega tardía)':{
        'order_id':'PRUEBA_002','order_purchase_timestamp':'2018-05-01T10:00:00',
        'order_approved_at':'2018-05-01T11:00:00','order_delivered_carrier_date':'2018-05-10T09:00:00',
        'order_delivered_customer_date':'2018-05-20T18:00:00','order_estimated_delivery_date':'2018-05-12T00:00:00',
        'estado_cliente':'SP','n_items':1,'n_vendedores':1,'precio_total':120.0,'flete_total':15.0,
        'peso_total_g':500.0,'fotos_promedio':2.0,'categoria':'beleza_saude',
        'estado_vendedor':'SP','tipo_pago':'credit_card','cuotas_max':3},
}


def inicio():
    titulo('📊 Olist | Inteligencia de reseñas','Proyecto final · Análisis predictivo, comentarios y segmentación')
    vendedores=cargar_csv('segmentacion_vendedores.csv')
    comentarios=cargar_csv('sentimiento_comentarios.csv')
    a,b,c=st.columns(3)
    a.metric('Vendedores registrados',f'{len(vendedores):,}' if vendedores is not None else 'Sin archivo')
    b.metric('Comentarios registrados',f'{len(comentarios):,}' if comentarios is not None else 'Sin archivo')
    c.metric('Módulos analíticos','3')
    st.subheader('¿Qué problema resolvemos?')
    st.write('Olist necesita priorizar la atención a pedidos con riesgo de recibir una reseña de 1 o 2 estrellas. La solución integra predicción de riesgo, análisis del sentimiento de los comentarios y agrupación de vendedores con características similares.')
    st.info('La capacidad operativa planteada en la guía es contactar al 10% de pedidos de mayor riesgo. Una predicción es una ayuda para priorizar, no una decisión automática sobre clientes o vendedores.')
    st.subheader('Componentes de la solución')
    st.markdown('**Predicción:** pipeline XGBoost servido mediante FastAPI en Render.  \n**Comentarios:** resultados del modelo de sentimiento contrastados con estrellas.  \n**Vendedores:** segmentación K-Means y propuestas de mejora por grupo.')
    st.markdown('**Extra (+5 puntos):** comparación de red neuronal Keras frente a XGBoost, con early stopping y curva de pérdida.')
    st.subheader('Disponibilidad de archivos')
    for nombre in ['segmentacion_vendedores.csv','sentimiento_comentarios.csv','perfiles_segmentos.csv','desacuerdos_analizados.csv','conclusion_comentarios.md','ficha_modelo.md']:
        st.write(('✅ ' if (ROOT/nombre).exists() else '⚪ ')+nombre)


def riesgo():
    titulo('⚠️ Predicción de riesgo','Consulta en vivo al servicio FastAPI desplegado en Render')
    st.write('Prueba un pedido válido, modifica sus datos o envía un JSON propio. La API verifica los campos y devuelve la probabilidad estimada de una mala reseña.')
    modo=st.radio('Modo de prueba',['Ejemplo guiado','JSON editable'],horizontal=True)
    ejemplo=st.selectbox('Caso de demostración',list(EJEMPLOS))
    datos=EJEMPLOS[ejemplo].copy()
    if modo=='Ejemplo guiado':
        with st.form('formulario_riesgo'):
            a,b=st.columns(2)
            with a:
                datos['order_id']=st.text_input('ID del pedido',datos['order_id'])
                datos['estado_cliente']=st.text_input('Estado del cliente',datos['estado_cliente'])
                datos['estado_vendedor']=st.text_input('Estado del vendedor',datos['estado_vendedor'])
                datos['categoria']=st.text_input('Categoría',datos['categoria'])
                datos['tipo_pago']=st.text_input('Tipo de pago',datos['tipo_pago'])
                datos['n_items']=st.number_input('Número de artículos',min_value=0,value=int(datos['n_items']))
                datos['n_vendedores']=st.number_input('Número de vendedores',min_value=0,value=int(datos['n_vendedores']))
                datos['cuotas_max']=st.number_input('Cuotas máximas',min_value=0.0,value=float(datos['cuotas_max']))
            with b:
                for campo,etiqueta in [('order_purchase_timestamp','Fecha de compra'),('order_approved_at','Fecha de aprobación'),('order_delivered_carrier_date','Fecha entregada al transportista'),('order_delivered_customer_date','Fecha de entrega al cliente'),('order_estimated_delivery_date','Fecha estimada de entrega')]:
                    datos[campo]=st.text_input(etiqueta+' (AAAA-MM-DDTHH:MM:SS)',datos[campo])
                for campo,etiqueta in [('precio_total','Precio total'),('flete_total','Flete total'),('peso_total_g','Peso total (g)'),('fotos_promedio','Promedio de fotos')]:
                    datos[campo]=st.number_input(etiqueta,min_value=0.0,value=float(datos[campo]))
            enviar=st.form_submit_button('Calcular riesgo',type='primary')
    else:
        st.caption('Para comprobar el error HTTP 422 puedes poner un precio negativo o eliminar un campo obligatorio.')
        texto=st.text_area('Cuerpo JSON',value=json.dumps(datos,indent=2,ensure_ascii=False),height=450)
        enviar=st.button('Enviar JSON a la API',type='primary')
        if enviar:
            try:datos=json.loads(texto)
            except json.JSONDecodeError as exc:
                st.error(f'El texto no es un JSON válido: {exc}')
                return
    if not enviar:return
    with st.spinner('Consultando API. Render puede tardar aproximadamente un minuto en despertar...'):
        try:
            respuesta=requests.post(api_url()+'/riesgo',json=datos,timeout=100)
        except requests.exceptions.Timeout:
            st.error('La API tardó demasiado en responder. Espera un momento y vuelve a intentarlo.');return
        except requests.exceptions.RequestException as exc:
            st.error('No se pudo conectar con la API. Comprueba el despliegue en Render.');st.caption(str(exc));return
    st.caption(f'Código HTTP: {respuesta.status_code}')
    if respuesta.status_code==422:
        st.warning('La API rechazó el pedido por un error de validación (HTTP 422).')
        try:st.json(respuesta.json())
        except ValueError:st.code(respuesta.text)
        return
    if not respuesta.ok:
        st.error(f'La API devolvió HTTP {respuesta.status_code}.')
        st.code(respuesta.text[:1500]);return
    try:resultado=respuesta.json()
    except ValueError:st.error('La respuesta no contiene JSON válido.');return
    prob=resultado.get('probabilidad')
    alto=resultado.get('riesgo_alto')
    if prob is None:
        st.error('La API respondió, pero no incluyó la probabilidad esperada.');st.json(resultado);return
    st.metric('Probabilidad estimada de mala reseña',f'{float(prob):.2%}')
    if alto is True:st.error('Riesgo alto: considerar seguimiento prioritario por atención al cliente.')
    elif alto is False:st.success('Riesgo no alto según el umbral configurado en la API.')
    else:st.info('La API no devolvió una clasificación de riesgo alto/bajo.')
    st.caption('La probabilidad no garantiza el resultado. El umbral se administra en FastAPI; no se modifica desde esta app.')
    with st.expander('Respuesta completa de la API'):st.json(resultado)


def comentarios():
    titulo('💬 Análisis de comentarios','Resultados del modelo de sentimiento comparados con las estrellas')
    df=cargar_csv('sentimiento_comentarios.csv')
    if df is None:aviso_archivo('sentimiento_comentarios.csv');return
    st.metric('Registros en archivo',f'{len(df):,}')
    st.caption('Columnas disponibles: '+', '.join(map(str,df.columns)))
    estrella=buscar(df,'review_score','estrellas','puntuacion','calificacion')
    sentimiento=buscar(df,'sentimiento_modelo','prediccion_sentimiento','etiqueta_sentimiento','sentimiento','sentiment_label','sentiment')
    if sentimiento:
        st.subheader('Distribución de sentimientos')
        conteo=df[sentimiento].fillna('Sin etiqueta').astype(str).value_counts().reset_index()
        conteo.columns=['Sentimiento','Comentarios']
        st.plotly_chart(px.bar(conteo,x='Sentimiento',y='Comentarios',color='Sentimiento'),use_container_width=True)
    else:st.warning('No se identificó una columna de sentimiento. Revisa los nombres de las columnas del CSV.')
    if estrella and sentimiento:
        st.subheader('Sentimiento frente a estrellas')
        tabla=pd.crosstab(df[estrella],df[sentimiento],dropna=False)
        st.dataframe(tabla,use_container_width=True)
        st.plotly_chart(px.imshow(tabla,text_auto=True,aspect='auto',labels={'x':'Sentimiento','y':'Estrellas','color':'Comentarios'}),use_container_width=True)
        st.caption('La tabla cruzada muestra cantidades; no implica que las estrellas y el texto siempre coincidan.')
    st.subheader('Coincidencia entre sentimiento y estrellas')
    coincidencia=buscar(df,'coincide','coincidencia','acuerdo','match')
    if coincidencia:
        serie=df[coincidencia].dropna().astype(str).str.strip().str.lower()
        validos=serie.isin(['true','false','0','1','si','sí','no'])
        if validos.all() and len(serie):
            porcentaje=serie.isin(['true','1','si','sí']).mean()
            st.metric('Coincidencia texto-estrellas',f'{porcentaje:.2%}')
            st.caption(f'Calculado a partir de {len(serie):,} registros con indicador de coincidencia.')
        else:
            st.warning('La columna de coincidencia contiene etiquetas no reconocidas; revisa su codificación.')
    elif estrella and sentimiento:
        mapa={'positive':'positive','positivo':'positive','negative':'negative','negativo':'negative','neutral':'neutral','neutro':'neutral'}
        etiquetas=df[sentimiento].astype(str).str.strip().str.lower().map(mapa)
        notas=pd.to_numeric(df[estrella],errors='coerce')
        esperadas=notas.map(lambda n: 'negative' if pd.notna(n) and n<=2 else ('neutral' if pd.notna(n) and n==3 else ('positive' if pd.notna(n) and n>=4 else None)))
        validos=etiquetas.notna() & esperadas.notna()
        if validos.any():
            st.metric('Coincidencia texto-estrellas',f'{(etiquetas[validos]==esperadas[validos]).mean():.2%}')
            st.caption('Regla de la rúbrica: 1–2 negativo, 3 neutro y 4–5 positivo.')
        else:
            st.info('No se pudo calcular la coincidencia con las etiquetas disponibles.')
    else:
        st.info('Faltan columnas para calcular la coincidencia entre texto y estrellas.')
    st.subheader('Cinco desacuerdos interpretados')
    desacuerdos=cargar_csv('desacuerdos_analizados.csv')
    if desacuerdos is not None:
        st.dataframe(desacuerdos,use_container_width=True)
        if len(desacuerdos)<5:st.warning('La rúbrica requiere al menos cinco desacuerdos analizados.')
    else:
        st.warning('Falta el archivo de cinco desacuerdos interpretados por el equipo. A continuación se muestran candidatos reales, NO interpretaciones realizadas.')
        if coincidencia and sentimiento and estrella:
            flags=df[coincidencia].astype(str).str.lower().isin(['false','0','no'])
            candidatos=df.loc[flags].head(10)
            if len(candidatos):st.dataframe(candidatos,use_container_width=True)

    st.subheader('¿Qué aporta el texto frente a las estrellas?')
    conclusion=archivo_texto('conclusion_comentarios.md')
    if conclusion:st.markdown(conclusion)
    else:st.warning('Pendiente: agregar conclusion_comentarios.md con la conclusión basada en los resultados del notebook.')
    with st.expander('Explorar comentarios'):
        st.dataframe(df.head(500),use_container_width=True)
        csv_descarga(df,'sentimiento_comentarios.csv','Descargar comentarios')


def segmentos():
    titulo('👥 Segmentación de vendedores','Agrupación K-Means · Parte 3B')
    df=cargar_csv('segmentacion_vendedores.csv')
    if df is None:aviso_archivo('segmentacion_vendedores.csv');return
    seg=buscar(df,'segmento','cluster','segment')
    if seg is None:st.error('El CSV no contiene la columna segmento.');return
    identificador=buscar(df,'seller_id')
    st.metric('Segmentos identificados',str(df[seg].nunique()))
    st.metric('Vendedores en archivo',f'{df[identificador].nunique():,}' if identificador else f'{len(df):,}')
    tam=df.groupby(seg,dropna=False).size().rename('Vendedores').reset_index()
    st.subheader('Tamaño de cada segmento')
    st.plotly_chart(px.bar(tam,x=seg,y='Vendedores',color=seg,text='Vendedores',title='Distribución de vendedores por segmento'),use_container_width=True)
    st.subheader('Perfiles cuantitativos por segmento')
    excluir={seg,identificador}
    numericas=[]
    trabajo=df.copy()
    for col in df.columns:
        if col in excluir:continue
        convertido=pd.to_numeric(df[col],errors='coerce')
        if convertido.notna().sum()>=max(5,int(len(df)*0.5)):
            trabajo[col]=convertido
            numericas.append(col)
    if numericas:
        elegidas=st.multiselect('Indicadores para comparar',numericas,default=numericas[:min(5,len(numericas))])
        if elegidas:
            perfiles=trabajo.groupby(seg)[elegidas].mean().round(3)
            st.dataframe(perfiles,use_container_width=True)
            largo=perfiles.reset_index().melt(id_vars=seg,var_name='Indicador',value_name='Promedio')
            st.plotly_chart(px.bar(largo,x=seg,y='Promedio',color='Indicador',barmode='group'),use_container_width=True)
            st.caption('Los indicadores tienen unidades diferentes: comparar principalmente un mismo indicador entre segmentos.')
    else:
        st.warning('Este CSV solo permite ver la asignación de vendedores a segmentos, no los perfiles cuantitativos. Necesitas exportar desde Colab las variables por vendedor (pedidos, ventas, categorías, tasa de retraso y flete relativo).')
        st.caption('No se pueden deducir estas métricas únicamente de seller_id y segmento.')
    st.subheader('Nombres y acciones por segmento')
    perfiles_csv=cargar_csv('perfiles_segmentos.csv')
    if perfiles_csv is not None:
        requeridas={'segmento','nombre','perfil','accion'}
        if not requeridas.issubset(perfiles_csv.columns):
            st.error('perfiles_segmentos.csv necesita las columnas: segmento, nombre, perfil, accion.')
        else:
            st.dataframe(perfiles_csv[['segmento','nombre','perfil','accion']],use_container_width=True)
            faltantes=set(df[seg].astype(str).unique())-set(perfiles_csv['segmento'].astype(str).unique())
            if faltantes:st.warning('Faltan perfiles de los segmentos: '+', '.join(sorted(faltantes)))
    else:
        st.warning('Falta perfiles_segmentos.csv. Los nombres y acciones deben provenir de la interpretación real del notebook; no se asignan automáticamente.')
    st.subheader('Explorador de vendedores')
    opciones=sorted(df[seg].dropna().unique(),key=str)
    elegido=st.selectbox('Filtrar segmento', ['Todos']+list(opciones))
    filtrado=df if elegido=='Todos' else df[df[seg]==elegido]
    st.dataframe(filtrado,use_container_width=True)
    csv_descarga(filtrado,'segmentacion_filtrada.csv','Descargar vendedores filtrados')
    st.caption('La justificación de k, el preprocesamiento y la interpretación del PCA deben mantenerse documentados en el notebook.')


def validacion():
    titulo('📈 Validación y resultados','Comparación de modelos y decisiones de selección')
    st.write('La evaluación temporal evita utilizar información posterior al corte al medir el rendimiento de un modelo. Los siguientes valores corresponden a los resultados compartidos del notebook.')
    a,b,c=st.columns(3)
    a.metric('Pedidos de entrenamiento (comparación principal)','57,065')
    b.metric('Pedidos para validación','15,461')
    c.metric('Precisión Top 10% · XGBoost','49.48%')
    resultados=pd.DataFrame({'Método':['Selección aleatoria','Regla del retraso','Regresión logística','XGBoost'],'Precisión Top 10%':[11.89,38.55,45.99,49.48]})
    st.plotly_chart(px.bar(resultados,x='Método',y='Precisión Top 10%',text_auto='.2f',title='Precisión entre el 10% de pedidos priorizados'),use_container_width=True)
    st.dataframe(resultados,use_container_width=True,hide_index=True)
    st.success('XGBoost supera a la regla del retraso en 10.93 puntos porcentuales y a la selección aleatoria en 37.59 puntos porcentuales.')
    st.write('En términos sencillos, de cada 10 pedidos priorizados por XGBoost, aproximadamente 5 recibieron una mala reseña en esta validación. Es una interpretación aproximada de la precisión Top 10%, no una garantía individual.')
    st.warning('Estas cifras no son el resultado de evaluar pedidos ocultos del docente. La generalización debe comprobarse por separado.')
    st.info('Revisar en el notebook la consistencia entre las fechas disponibles al momento de predecir, la ausencia de fuga de datos y el umbral utilizado por la API.')



def plus():
    titulo('🧠 Plus · Red neuronal vs. XGBoost', 'Extra de la rúbrica (+5 puntos) · Red neuronal densa Keras con early stopping')
    st.write('Se entrenó una red neuronal densa con las mismas variables procesadas y la misma muestra de validación temporal utilizada para la comparación con XGBoost. Los valores siguientes provienen de las salidas ejecutadas del notebook, no de una nueva ejecución en Streamlit.')
    a,b,c=st.columns(3)
    a.metric('Pedidos de validación','15,461')
    b.metric('Características procesadas','140')
    c.metric('Mejor época de Keras','5 de 10')
    st.subheader('Precisión entre el 10% de pedidos con mayor riesgo')
    comparacion=pd.DataFrame({'Modelo':['Red neuronal Keras','XGBoost'],'Precisión Top 10% (%)':[48.67,48.93]})
    st.plotly_chart(px.bar(comparacion,x='Modelo',y='Precisión Top 10% (%)',color='Modelo',text_auto='.2f',range_y=[0,60]),use_container_width=True)
    st.dataframe(comparacion,hide_index=True,use_container_width=True)
    st.success('XGBoost supera a Keras por 0.26 puntos porcentuales en esta ejecución comparable.')
    st.caption('Nota de trazabilidad: otra sección anterior del notebook reporta 49.48% para XGBoost. Para comparar Keras vs. XGBoost se usan los valores de la celda del extra, que evaluó ambos modelos sobre la misma muestra.')
    st.subheader('Curva de pérdida por época')
    st.caption('Valores redondeados mostrados por el registro de entrenamiento de Keras en Colab. No son mediciones realizadas por esta app.')
    # Pérdidas redondeadas de las 10 épocas, extraídas del registro del notebook.
    perdidas=pd.DataFrame({'Época':list(range(1,11)),
                            'Entrenamiento':[0.3591,0.3265,0.3215,0.3177,0.3147,0.3125,0.3107,0.3090,0.3074,0.3058],
                            'Validación':[0.3109,0.3068,0.3051,0.3044,0.3039,0.3040,0.3043,0.3047,0.3055,0.3061]})
    st.plotly_chart(px.line(perdidas.melt(id_vars='Época',var_name='Conjunto',value_name='Pérdida'),x='Época',y='Pérdida',color='Conjunto',markers=True),use_container_width=True)
    st.info('La menor pérdida de validación fue 0.3039 en la época 5. Aunque el error de entrenamiento continuó disminuyendo, el de validación dejó de mejorar; early stopping detuvo el entrenamiento tras 10 épocas y recuperó los mejores pesos.')
    st.subheader('Arquitectura y conclusión')
    st.write('Red neuronal densa: entrada de 140 características → capa de 64 neuronas ReLU → capa de 32 neuronas ReLU → salida sigmoide. Optimizador Adam, pérdida binary crossentropy, lote 256 y paciencia 5 para early stopping.')
    st.write('Los resultados son cercanos. En esta ejecución XGBoost fue ligeramente superior, por lo que sigue siendo razonable mantenerlo en la API. En datos tabulares, los árboles potenciados pueden capturar interacciones sin requerir la misma adaptación que una red densa.')
    st.warning('La curva utiliza las 10 épocas del registro de Colab, redondeadas a cuatro decimales; para reproducir valores de precisión completa se debe exportar historial_nn.history.')


def ficha():
    titulo('📄 Ficha del modelo','Información para decidir cuándo y cómo utilizar la predicción')
    texto=archivo_texto('ficha_modelo.md')
    if texto:
        st.markdown(texto)
        st.download_button('Descargar ficha',texto,'ficha_modelo.md','text/markdown')
        return
    st.warning('Para la entrega debe añadirse ficha_modelo.md al repositorio. Esta vista provisional resume lo que está respaldado por la información disponible.')
    st.subheader('1. Para qué sirve y para qué no')
    st.write('Estima el riesgo de una reseña de 1 o 2 estrellas y ayuda a priorizar contactos. No debe usarse para sancionar vendedores ni para tomar decisiones irreversibles.')
    st.subheader('2. Datos')
    st.write('Proyecto basado en datos de Olist. Los resultados del notebook muestran 15,461 pedidos de validación temporal. El número de entrenamiento depende de la ejecución comparada (en el extra se registraron 61,843). Falta consolidar el periodo exacto, exclusiones y versión final en ficha_modelo.md.')
    st.subheader('3. Qué tan bien funciona')
    st.write('Precisión Top 10% en validación temporal: XGBoost 49.48%, regresión logística 45.99%, regla del retraso 38.55% y azar 11.89%. Aproximadamente 5 de cada 10 pedidos priorizados por XGBoost presentaron el resultado de interés.')
    st.subheader('4. Límites conocidos')
    st.write('Puede fallar ante cambios en las condiciones de entrega, datos faltantes, categorías poco representadas y diferencias entre la muestra histórica y pedidos futuros.')
    st.subheader('5. A quién afecta un error')
    st.write('Un falso positivo puede provocar una llamada innecesaria. Un falso negativo puede dejar sin atención prioritaria a un cliente insatisfecho. Penalizar vendedores por predicciones o retrasos ajenos a su control sería un uso inapropiado.')
    st.subheader('6. Recomendación y supervisión')
    st.write('Usar el modelo como apoyo a un equipo humano, revisar periódicamente precisión y errores por grupos y actualizarlo cuando cambien los datos. Definir con el responsable operativo una frecuencia formal de revisión.')

paginas=[
    st.Page(inicio,title='Inicio',icon='🏠'),
    st.Page(riesgo,title='Predicción de riesgo',icon='⚠️'),
    st.Page(comentarios,title='Análisis de comentarios',icon='💬'),
    st.Page(segmentos,title='Segmentación 3B',icon='👥'),
    st.Page(validacion,title='Validación y resultados',icon='📈'),
    st.Page(plus,title='Plus · Keras vs. XGBoost',icon='🧠'),
    st.Page(ficha,title='Ficha del modelo',icon='📄'),
]
st.navigation(paginas).run()
st.divider()
st.caption('Proyecto final Olist · Resultados académicos · Uso con supervisión humana')
