
import json
import time
from requests.exceptions import Timeout, ConnectionError as APIConnectionError, RequestException
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title='Olist | Inteligencia de reseñas', page_icon='📊', layout='wide')
ROOT = Path(__file__).resolve().parent
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1200px}h1,h2{letter-spacing:-.025em}[data-testid="stMetric"]{background:rgba(128,128,128,.06);padding:18px;border-radius:12px;border:1px solid rgba(128,128,128,.15)}div.stButton>button[kind=primary]{border-radius:10px}section[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.12)}</style>''', unsafe_allow_html=True)

@st.cache_data
def leer(nombre):
    return pd.read_csv(ROOT / nombre, encoding='utf-8-sig')

def columna(df, *opciones):
    normal = {str(c).lower().strip(): c for c in df.columns}
    for o in opciones:
        if o.lower() in normal:
            return normal[o.lower()]
    for c in df.columns:
        if any(o.lower() in str(c).lower() for o in opciones):
            return c
    return None

def datos():
    try:
        return leer('segmentacion_vendedores.csv'), leer('sentimiento_comentarios.csv')
    except Exception as e:
        st.error(f'No se pudieron leer los CSV del repositorio: {e}')
        st.stop()

def titulo(t, sub):
    st.title(t)
    st.caption(sub)

def inicio():
    titulo('📊 Olist | Inteligencia de reseñas',
           'Proyecto final · Riesgo, sentimiento y segmentación de vendedores')
    v, c = datos()
    a, b, d = st.columns(3)
    a.metric('Registros de segmentación', f'{len(v):,}')
    b.metric('Comentarios analizados', f'{len(c):,}')
    d.metric('Módulos analíticos', '3')
    st.markdown('### ¿Qué resuelve este proyecto?')
    st.write('Integra un modelo de riesgo de malas reseñas (XGBoost), análisis de sentimientos de comentarios y segmentación de vendedores (opción 3B).')
    st.info('Los indicadores muestran los registros de los archivos cargados; no representan necesariamente el total del conjunto Olist.')
    st.markdown('### Flujo de trabajo')
    st.write('**Pedido → API de riesgo → probabilidad y alerta** · **Comentarios → sentimiento vs. estrellas** · **Vendedores → segmentos y perfiles**')

def riesgo():
    titulo('⚠️ Riesgo de una mala reseña',
           'Predicción en vivo mediante FastAPI y XGBoost desplegados en Render')
    try:
        api = st.secrets.get('API_URL', '')
    except Exception:
        api = ''
    if not api:
        st.warning('Configura API_URL en los Secrets de Streamlit Community Cloud para activar las predicciones.')
        st.code('API_URL = "https://api-olist-riesgo.onrender.com"', language='toml')
        return

    ejemplo = {
        'order_id': 'pedido_prueba_001',
        'order_purchase_timestamp': '2018-05-10T10:00:00',
        'order_approved_at': '2018-05-10T11:00:00',
        'order_delivered_carrier_date': '2018-05-12T09:00:00',
        'order_delivered_customer_date': '2018-05-20T14:00:00',
        'order_estimated_delivery_date': '2018-05-18T00:00:00',
        'estado_cliente': 'SP',
        'n_items': 2,
        'n_vendedores': 1,
        'precio_total': 150.0,
        'flete_total': 20.0,
        'peso_total_g': 1200.0,
        'fotos_promedio': 3.0,
        'categoria': 'beleza_saude',
        'estado_vendedor': 'SP',
        'tipo_pago': 'credit_card',
        'cuotas_max': 3
    }

    st.markdown('#### Selecciona o edita un pedido')
    if 'pedidos_ejemplo.csv' in [p.name for p in ROOT.iterdir()]:
        try:
            muestra = leer('pedidos_ejemplo.csv')
            idx = st.selectbox(
                'Pedido de muestra',
                range(len(muestra)),
                format_func=lambda i: str(muestra.iloc[i].get('order_id', f'Pedido {i+1}'))
            )
            candidato = muestra.iloc[idx].where(muestra.iloc[idx].notna(), None).to_dict()
            ejemplo = {
                k: (v.isoformat() if hasattr(v, 'isoformat')
                    else v.item() if hasattr(v, 'item') else v)
                for k, v in candidato.items()
            }
        except Exception as e:
            st.warning(f'No se pudo leer la muestra: {e}')
    else:
        st.caption('Puedes editar el pedido de ejemplo. Para seleccionar pedidos reales, agrega pedidos_ejemplo.csv al repositorio.')

    with st.expander('¿Qué datos se envían al modelo?', expanded=False):
        st.write('El pedido contiene fechas, estado del cliente, productos, vendedores, precios, flete y pago. No modifiques los nombres de las variables.')

    texto = st.text_area(
        'Datos del pedido (JSON con las 17 columnas del contrato)',
        json.dumps(ejemplo, indent=2, ensure_ascii=False),
        height=420
    )
    st.caption('La respuesta es una estimación estadística; no una certeza sobre la opinión del cliente.')

    if st.button('Calcular riesgo', type='primary'):
        try:
            payload = json.loads(texto)
            if not isinstance(payload, dict):
                raise ValueError('El cuerpo debe ser un objeto JSON.')

            obligatorios = [
                'order_id', 'order_purchase_timestamp',
                'order_estimated_delivery_date', 'estado_cliente',
                'n_items', 'n_vendedores', 'precio_total',
                'flete_total', 'fotos_promedio', 'categoria',
                'estado_vendedor', 'tipo_pago', 'cuotas_max'
            ]
            faltantes = [x for x in obligatorios if x not in payload]
            if faltantes:
                raise ValueError('Faltan campos obligatorios: ' + ', '.join(faltantes))

            with st.spinner('Consultando API (Render puede tardar en activarse)...'):
                r = requests.post(
                    api.rstrip('/') + '/riesgo',
                    json=payload,
                    timeout=120
                )

            if r.status_code == 422:
                st.error('El pedido no cumple el contrato de la API (HTTP 422). Comprueba campos, fechas y valores.')
                with st.expander('Detalle de validación'):
                    try:
                        st.json(r.json())
                    except ValueError:
                        st.code(r.text[:1500])
                return

            if r.status_code != 200:
                st.error(f'La API respondió HTTP {r.status_code}. Intenta de nuevo o revisa el estado de Render.')
                with st.expander('Detalle técnico'):
                    st.code(r.text[:1500])
                return

            resultado = r.json()
            p = resultado.get('probabilidad')
            alto = resultado.get('riesgo_alto')

            if p is None or not isinstance(alto, bool):
                st.error('La API devolvió una respuesta incompleta.')
                st.json(resultado)
                return

            if not 0 <= float(p) <= 1:
                raise ValueError('La API devolvió una probabilidad fuera del intervalo [0,1].')

            st.metric('Probabilidad estimada de mala reseña', f'{float(p):.2%}')
            st.progress(min(max(float(p), 0), 1))

            if alto:
                st.error('Riesgo alto según el umbral del modelo: priorizar seguimiento.')
            else:
                st.success('Riesgo no alto según el umbral del modelo.')

            st.json(resultado)

        except (ValueError, json.JSONDecodeError) as e:
            st.error(f'JSON inválido: {e}')
        except Timeout:
            st.error('La API tardó demasiado en responder. En el plan gratuito de Render puede necesitar tiempo para activarse. Espera un momento y vuelve a intentar.')
        except APIConnectionError:
            st.error('No se pudo conectar con la API. Verifica tu conexión y que el servicio de Render esté activo.')
        except RequestException as e:
            st.error('Ocurrió un problema de comunicación con la API.')
            with st.expander('Detalle técnico'):
                st.code(str(e))
        except Exception as e:
            st.error(f'Error inesperado: {e}')

def comentarios():
    titulo('💬 Sentimiento de los comentarios',
           'Parte 4 · Comparación entre calificación y modelo de lenguaje')
    _, df = datos()
    estrellas = columna(df, 'review_score')
    sent = columna(df, 'sentimiento_modelo')

    a, b = st.columns(2)
    a.metric('Comentarios procesados', f'{len(df):,}')

    if sent:
        distribucion = (
            df[sent].fillna('Sin clasificar').astype(str)
            .value_counts().rename_axis('Sentimiento')
            .reset_index(name='Cantidad')
        )
        fig = px.bar(
            distribucion, x='Sentimiento', y='Cantidad',
            color='Sentimiento', title='Distribución de sentimiento'
        )
        st.plotly_chart(fig, use_container_width=True)

    if estrellas and sent:
        base = df[[estrellas, sent]].dropna().copy()
        tabla = pd.crosstab(base[estrellas], base[sent])

        st.subheader('Tabla cruzada: estrellas × sentimiento')
        st.dataframe(tabla, use_container_width=True)

        st.caption('Porcentaje de cada fila (dentro de cada calificación de estrellas)')
        porcentajes = (
            tabla.div(
                tabla.sum(axis=1).replace(0, float('nan')),
                axis=0
            ).mul(100).round(1).fillna(0)
        )
        st.dataframe(porcentajes, use_container_width=True)

        fig = px.imshow(
            tabla, text_auto=True, aspect='auto',
            labels={
                'x': 'Sentimiento',
                'y': 'Estrellas',
                'color': 'Comentarios'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('#### Coincidencia y desacuerdos')
        st.caption('Criterio operativo: 1–2 estrellas = negativo, 3 = neutral, 4–5 = positivo. Verifica que las etiquetas del modelo correspondan a estas categorías.')

        etiquetas = {
            1: 'negativo', 2: 'negativo',
            3: 'neutral', 4: 'positivo', 5: 'positivo'
        }
        esperado = pd.to_numeric(
            base[estrellas], errors='coerce'
        ).map(etiquetas)
        observado = base[sent].astype(str).str.lower().str.strip()
        validos = observado.isin(
            ['positivo', 'negativo', 'neutral']
        ) & esperado.notna()

        if validos.any():
            coinciden = (esperado[validos] == observado[validos])
            b.metric('Coincidencia estrellas/sentimiento', f'{coinciden.mean():.1%}')

            idx = base.loc[validos].index[~coinciden]
            st.write(f'Desacuerdos identificados: **{len(idx)}**')
            desacuerdos = df.loc[idx].copy()
            st.dataframe(desacuerdos.head(30), use_container_width=True)

            st.download_button(
                'Descargar desacuerdos',
                desacuerdos.to_csv(index=False).encode('utf-8-sig'),
                'desacuerdos_olist.csv', 'text/csv'
            )

            if len(idx) >= 5:
                st.info('La rúbrica exige leer y clasificar cinco casos como error del modelo o del cliente. Añade esas interpretaciones verificadas en desacuerdos_analizados.csv.')
                ruta = ROOT / 'desacuerdos_analizados.csv'
                if ruta.exists():
                    try:
                        revisados = leer('desacuerdos_analizados.csv')
                        st.subheader('Cinco casos revisados por el equipo')
                        st.dataframe(revisados, use_container_width=True)
                        if len(revisados) < 5:
                            st.warning('Faltan casos revisados para llegar a cinco.')
                    except Exception as e:
                        st.warning(f'No se pudo cargar el análisis de casos: {e}')
            else:
                st.warning('Hay menos de cinco desacuerdos bajo este criterio.')
        else:
            st.warning('Las etiquetas no son positivo/negativo/neutral. Ajusta su equivalencia antes de calcular coincidencia.')
    else:
        st.warning('No se identificaron review_score y sentimiento_modelo. Revisa las columnas del CSV.')

    st.subheader('Explorar comentarios')
    st.dataframe(df, use_container_width=True)
    st.download_button(
        'Descargar resultados',
        df.to_csv(index=False).encode('utf-8-sig'),
        'sentimiento_comentarios.csv', 'text/csv'
    )

    ruta_conclusion = ROOT / 'conclusion_comentarios.md'
    if ruta_conclusion.exists():
        st.subheader('¿Qué aporta el texto que no muestran las estrellas?')
        st.markdown(ruta_conclusion.read_text(encoding='utf-8'))
    else:
        st.info('Pendiente de documentar con evidencia: conclusión sobre qué aporta el texto frente a las estrellas (conclusion_comentarios.md).')

def segmentos():
    titulo('👥 Segmentación de vendedores',
           'Parte 3B · Exploración de segmentos obtenidos mediante KMeans')
    df, _ = datos()
    seg = columna(df, 'segmento', 'cluster', 'grupo')

    if not seg:
        st.warning('No se detectó una columna de segmento.')
        st.write('Columnas:', df.columns.tolist())
        st.dataframe(df)
        return

    st.metric('Segmentos identificados', df[seg].nunique())

    conteo = (
        df[seg].fillna('Sin asignar').astype(str)
        .value_counts().rename_axis('Segmento')
        .reset_index(name='Vendedores')
    )
    st.plotly_chart(
        px.bar(
            conteo, x='Segmento', y='Vendedores',
            color='Segmento', title='Tamaño de cada segmento'
        ),
        use_container_width=True
    )

    numericas = [
        c for c in df.select_dtypes(include='number').columns
        if c != seg and df[c].notna().any()
    ]

    st.subheader('Perfiles cuantitativos por segmento')
    if numericas:
        cols = st.multiselect(
            'Variables para comparar',
            numericas,
            default=numericas[:min(5, len(numericas))]
        )
        if cols:
            perfiles = df.groupby(seg, dropna=False)[cols].mean().round(2)
            st.dataframe(perfiles, use_container_width=True)
            grafico = perfiles.reset_index().melt(
                id_vars=seg,
                var_name='Variable',
                value_name='Promedio'
            )
            st.plotly_chart(
                px.bar(
                    grafico, x=seg, y='Promedio',
                    color='Variable', barmode='group',
                    title='Promedios por segmento'
                ),
                use_container_width=True
            )
    else:
        st.warning('No se encontraron variables numéricas para calcular perfiles.')

    st.subheader('Nombres y acciones por segmento')
    st.info('La rúbrica pide nombres interpretables y acciones para cada segmento. Añade las conclusiones REALES del notebook en un archivo perfiles_segmentos.csv con columnas: segmento,nombre,perfil,accion.')

    if (ROOT / 'perfiles_segmentos.csv').exists():
        perfiles_texto = leer('perfiles_segmentos.csv')
        st.dataframe(perfiles_texto, use_container_width=True)
        requeridas = {'segmento', 'nombre', 'perfil', 'accion'}
        if not requeridas.issubset(perfiles_texto.columns):
            st.warning('El archivo de perfiles debe incluir: segmento, nombre, perfil y accion.')
        elif perfiles_texto['segmento'].nunique() < df[seg].nunique():
            st.warning('No hay perfiles documentados para todos los segmentos.')

    st.subheader('Detalle de vendedores')
    st.dataframe(df, use_container_width=True)
    st.download_button(
        'Descargar segmentación',
        df.to_csv(index=False).encode('utf-8-sig'),
        'segmentacion_vendedores.csv', 'text/csv'
    )

def ficha():
    titulo('📄 Ficha del modelo',
           'Documentación técnica y límites de uso')
    ruta = ROOT / 'ficha_modelo.md'

    if ruta.exists():
        st.markdown(ruta.read_text(encoding='utf-8'))
    else:
        st.warning('Pendiente: agregar ficha_modelo.md con los seis puntos de la sección 11 de la guía. No se muestran métricas no verificadas.')
        st.write('**Información confirmada:** API FastAPI; algoritmo XGBoost; objetivo: estimar riesgo de mala reseña; umbral utilizado en la API: 0.2704 (verificar que coincida con la versión final del entrenamiento).')
        st.write('**Por completar con el notebook:** datos, validación temporal, métricas, comparación contra línea base, variables, limitaciones y decisiones de uso.')

paginas = [
    st.Page(inicio, title='Inicio', icon='🏠'),
    st.Page(riesgo, title='Predicción de riesgo', icon='⚠️'),
    st.Page(comentarios, title='Análisis de comentarios', icon='💬'),
    st.Page(segmentos, title='Segmentación 3B', icon='👥'),
    st.Page(ficha, title='Ficha del modelo', icon='📄'),
]

st.navigation(paginas).run()
st.divider()
st.caption('Proyecto final Olist · Resultados académicos; las probabilidades no sustituyen una decisión humana.')
