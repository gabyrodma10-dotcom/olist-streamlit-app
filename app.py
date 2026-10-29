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
    titulo('⚠️ Predicción de riesgo', 'Simulador sencillo de pedidos · conectado a FastAPI')
    st.write('**¿Qué queremos saber?** Si un pedido podría recibir una reseña de 1 o 2 estrellas.')
    st.info('Empieza con un caso preparado. Solo necesitas pulsar **Calcular riesgo**. Después puedes cambiar los días de entrega y comparar.')
    caso=st.radio(
        '1. Elige un escenario',
        ['📦 Entrega puntual', '🚚 Entrega con retraso'],
        horizontal=True,
        help='Son ejemplos de demostración, no pedidos reales.'
    )
    clave='Riesgo bajo (entrega puntual)' if caso.startswith('📦') else 'Riesgo alto (entrega tardía)'
    datos=EJEMPLOS[clave].copy()

    fecha_estimada=pd.to_datetime(datos['order_estimated_delivery_date'])
    fecha_real=pd.to_datetime(datos['order_delivered_customer_date'])
    diferencia=(fecha_real.normalize()-fecha_estimada.normalize()).days

    st.subheader('2. Revisa o cambia el pedido')
    with st.form('simulador_riesgo'):
        a,b,c=st.columns(3)
        a.metric('Fecha prometida', fecha_estimada.strftime('%d/%m/%Y'))
        b.metric('Entrega del ejemplo', fecha_real.strftime('%d/%m/%Y'))
        c.metric('Días respecto al plazo',f'{diferencia:+d}',help='Negativo: antes de lo previsto. Positivo: después.')
        dias=st.slider(
            '¿Cuántos días antes o después de la fecha prometida llegó el pedido?',
            min_value=-10,max_value=20,value=int(diferencia),
            help='-3 significa tres días antes; +8 significa ocho días de retraso.'
        )
        fecha_nueva=fecha_estimada+pd.Timedelta(days=dias)
        datos['order_delivered_customer_date']=fecha_nueva.strftime('%Y-%m-%dT15:00:00')
        if dias>0:
            st.warning(f'En esta simulación el pedido llega **{dias} días tarde**.')
        elif dias<0:
            st.success(f'En esta simulación el pedido llega **{abs(dias)} días antes**.')
        else:
            st.success('En esta simulación el pedido llega **justo en la fecha prometida**.')
        with st.expander('⚙️ Ajustes avanzados (opcional): ver y editar los 17 campos'):
            st.caption('No necesitas tocar estos campos para la demostración básica.')
            izq,der=st.columns(2)
            with izq:
                for campo,etiqueta in [('order_id','ID del pedido'),('estado_cliente','Estado del cliente'),('estado_vendedor','Estado del vendedor'),('categoria','Categoría'),('tipo_pago','Tipo de pago')]:
                    datos[campo]=st.text_input(etiqueta,value=str(datos[campo]))
                for campo,etiqueta in [('n_items','Número de artículos'),('n_vendedores','Número de vendedores'),('cuotas_max','Cuotas máximas')]:
                    datos[campo]=st.number_input(etiqueta,min_value=0,value=int(datos[campo]),step=1)
            with der:
                for campo,etiqueta in [('order_purchase_timestamp','Fecha de compra'),('order_approved_at','Fecha de aprobación'),('order_delivered_carrier_date','Entrega al transportista'),('order_delivered_customer_date','Entrega real'),('order_estimated_delivery_date','Entrega prometida')]:
                    datos[campo]=st.text_input(etiqueta,value=str(datos[campo]))
                for campo,etiqueta in [('precio_total','Precio total'),('flete_total','Costo de envío'),('peso_total_g','Peso en gramos'),('fotos_promedio','Fotos promedio')]:
                    datos[campo]=st.number_input(etiqueta,min_value=0.0,value=float(datos[campo]))
        enviar=st.form_submit_button('🔎 Calcular riesgo del pedido',type='primary',use_container_width=True)

    with st.expander('🧪 Prueba técnica de la API (JSON editable)'):
        st.caption('Para la defensa: permite enviar un JSON manual y comprobar errores HTTP 422.')
        texto=st.text_area('JSON del pedido',value=json.dumps(datos,indent=2,ensure_ascii=False),height=250)
    if st.button('Enviar JSON técnico',key='json_tecnico'):
        try:
            datos=json.loads(texto)
            enviar=True
        except json.JSONDecodeError as exc:
            st.error(f'JSON inválido: {exc}')
            return
    if not enviar:return
    with st.spinner('Consultando el modelo. Render puede tardar hasta un minuto en activarse...'):
        try:
            respuesta=requests.post(api_url()+'/riesgo',json=datos,timeout=100)
        except requests.exceptions.Timeout:
            st.error('La API tardó demasiado. Intenta de nuevo en un momento.');return
        except requests.exceptions.RequestException:
            st.error('No se pudo conectar con la API de Render.');return
    if respuesta.status_code==422:
        st.warning('La API rechazó los datos (HTTP 422). Revisa los campos.')
        try:st.json(respuesta.json())
        except ValueError:st.code(respuesta.text)
        return
    if not respuesta.ok:
        st.error(f'La API devolvió HTTP {respuesta.status_code}.')
        st.code(respuesta.text[:1000]);return
    try:resultado=respuesta.json()
    except ValueError:
        st.error('La API no devolvió un JSON válido.');return
    prob=resultado.get('probabilidad')
    alto=resultado.get('riesgo_alto')
    if prob is None:
        st.error('La API no devolvió la probabilidad esperada.');st.json(resultado);return
    st.subheader('3. Resultado de la predicción')
    st.metric('Probabilidad estimada de reseña negativa',f'{float(prob):.2%}')
    if alto is True:
        st.error('**Riesgo alto:** este pedido sería candidato a seguimiento prioritario.')
    elif alto is False:
        st.success('**Riesgo no alto:** el pedido no supera el umbral de priorización.')
    else:
        st.info('La API no incluyó una clasificación alto/no alto.')
    st.caption('El modelo considera los 17 campos, no solo el retraso. La clasificación usa el umbral del 10% de pedidos con mayor riesgo, no un corte de 50%.')
    st.caption('Estos ejemplos usan datos de entrega real: no representan una predicción antes de que el pedido sea entregado.')
    with st.expander('Ver respuesta técnica completa'):
        st.json(resultado)

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
    st.caption('Perfiles descriptivos calculados automáticamente a partir de los indicadores reales del CSV. Los nombres son propuestas interpretativas, no etiquetas aprendidas por K-Means.')
    campos=['total_pedidos','ventas_totales','n_categorias','tasa_retraso','flete_relativo']
    faltan=[c for c in campos if c not in trabajo.columns]
    if faltan:
        st.warning('No se pueden interpretar todos los segmentos: faltan indicadores '+', '.join(faltan))
    else:
        base=trabajo.groupby(seg)[campos].mean()
        globales=trabajo[campos].median()
        registros=[]
        for grupo, fila in base.iterrows():
            volumen='alto' if fila['total_pedidos']>globales['total_pedidos'] else 'moderado o bajo'
            ventas='altas' if fila['ventas_totales']>globales['ventas_totales'] else 'moderadas o bajas'
            retrasos='elevados' if fila['tasa_retraso']>globales['tasa_retraso'] else 'contenidos'
            flete='elevado' if fila['flete_relativo']>globales['flete_relativo'] else 'contenido'
            if retrasos=='elevados' and volumen=='alto':
                nombre='Alto volumen con retos de entrega'
                accion='Priorizar revisión logística, tiempos de despacho y seguimiento de pedidos.'
            elif retrasos=='elevados':
                nombre='Vendedores con oportunidad logística'
                accion='Investigar causas de retrasos y definir un plan de mejora del cumplimiento.'
            elif volumen=='alto' and ventas=='altas':
                nombre='Vendedores consolidados'
                accion='Mantener buenas prácticas y explorar acciones para fortalecer su crecimiento.'
            elif flete=='elevado':
                nombre='Vendedores con flete relativo alto'
                accion='Revisar costos y opciones de envío antes de proponer optimizaciones.'
            else:
                nombre='Vendedores de operación moderada'
                accion='Monitorear resultados y explorar oportunidades de desarrollo comercial.'
            perfil=(f"Pedidos: {fila['total_pedidos']:.1f} ({volumen}); "
                    f"ventas: {fila['ventas_totales']:.2f} ({ventas}); "
                    f"categorías: {fila['n_categorias']:.1f}; "
                    f"tasa de retraso: {fila['tasa_retraso']:.3f} ({retrasos}); "
                    f"flete relativo: {fila['flete_relativo']:.3f} ({flete}).")
            registros.append({'Segmento':grupo,'Nombre propuesto':nombre,'Perfil basado en promedios':perfil,'Acción sugerida':accion})
        tabla_perfiles=pd.DataFrame(registros)
        st.dataframe(tabla_perfiles,use_container_width=True,hide_index=True)
        st.info('Los criterios alto/bajo se comparan con la mediana de los vendedores del archivo. '
                'Revisa estos nombres y acciones contra la interpretación de tu notebook antes de entregar: '
                'son sugerencias automatizadas, no conclusiones validadas por el entrenamiento.')
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
