# Ficha del modelo — Olist: del pedido a la reseña

**Proyecto:** Predicción del riesgo de reseñas negativas en pedidos de comercio electrónico  
**Modelo seleccionado:** XGBoost  
**Variable objetivo:** Reseña negativa (1 o 2 estrellas)  
**Métrica principal:** Precisión Top 10%

## 1. ¿Para qué sirve el modelo y para qué no?

El modelo permite identificar los pedidos con mayor probabilidad de recibir una reseña negativa, con el propósito de ayudar al equipo de atención al cliente a decidir cuáles casos revisar primero.

Por ejemplo, si el equipo solo puede dar seguimiento a 10 de cada 100 pedidos, el modelo permite seleccionar los 10 que presentan mayor riesgo estimado, en lugar de elegirlos al azar.

**¿Para qué sí puede utilizarse?**

- Priorizar el seguimiento de pedidos que ya fueron entregados.
- Identificar casos que podrían requerir atención.
- Apoyar la distribución del trabajo de un equipo con capacidad limitada.
- Complementar el análisis de problemas logísticos y satisfacción del cliente.

**¿Para qué no debe utilizarse?**

No debe presentarse como una herramienta para anticipar reseñas negativas antes de la entrega, porque utiliza información de la entrega real.

Tampoco debe emplearse para sancionar automáticamente a vendedores, garantizar que un cliente dejará una mala reseña o sustituir el criterio del personal de atención.

**Recomendación de uso:** utilizarlo como herramienta de apoyo a las decisiones, con supervisión humana.

## 2. ¿De dónde vienen los datos, qué período cubren y qué se excluyó?

Se analizaron **77,304 pedidos históricos de Olist**, correspondientes al período comprendido entre el **15 de septiembre de 2016 y el 31 de agosto de 2018**, según la fecha de compra (`order_purchase_timestamp`).

Para la validación temporal, se reservó el 20% de los pedidos más recientes, cuyo período comienza el **22 de marzo de 2018 a las 10:15:19**. Esta separación permitió evaluar el modelo respetando el orden cronológico de las compras.

| Conjunto | Cantidad de pedidos |
|---|---:|
| Entrenamiento temporal inicial | 61,843 |
| Validación temporal | 15,461 |
| Entrenamiento temporal estricto | 57,065 |

Se realizaron dos exclusiones adicionales en el entrenamiento:

- **4,010 pedidos** cuya entrega ocurrió después del inicio del período de validación.
- **768 pedidos adicionales** cuyas reseñas no estaban confirmadas antes de esa fecha.

Estas exclusiones permitieron evaluar el modelo de manera más conservadora, reduciendo el riesgo de entrenarlo con información que todavía no habría estado disponible.

El modelo utiliza características de los pedidos y sus entregas, como días de retraso, duración de entrega, cantidades de artículos, costos de envío, precios y categorías.

**Para evitar fuga de información**, no se utilizaron como variables predictoras las estrellas, el texto ni las fechas de las reseñas. Las estrellas se utilizaron únicamente para identificar si el resultado fue una reseña negativa.

Después de la evaluación, el modelo final se entrenó con los **77,304 pedidos históricos** y se guardó como `modelo_riesgo.joblib`.

## 3. ¿Qué tan bien funciona el modelo?

La evaluación se realizó respetando el orden temporal de las compras, utilizando 57,065 pedidos para entrenamiento estricto y 15,461 para validación.

Se utilizó la métrica **Precisión Top 10%**, que mide cuántos pedidos realmente recibieron una reseña negativa entre el 10% que el modelo consideró de mayor riesgo.

### Resultados de la validación

| Método | Precisión Top 10% |
|---|---:|
| Selección aleatoria | 11.89% |
| Regla basada en retrasos | 38.55% |
| Regresión logística | 45.99% |
| **XGBoost** | **49.22%** |

**¿Qué significan estos resultados en palabras sencillas?**

Imaginemos que el equipo solo puede contactar a 10 clientes:

- Si los selecciona al azar, encontraría aproximadamente **1 cliente** que dejará una reseña negativa.
- Si utiliza únicamente los días de retraso, encontraría aproximadamente **4 clientes**.
- Si utiliza XGBoost, encontraría aproximadamente **5 clientes**.

Este ejemplo es una aproximación basada en los porcentajes de validación, no una garantía para cada grupo de 10 pedidos.

XGBoost supera a la selección aleatoria por **37.33 puntos porcentuales** y a la regla del retraso por **10.67 puntos porcentuales**.

También supera a la regresión logística por **3.23 puntos porcentuales**.

**Conclusión:** XGBoost fue seleccionado porque permitió identificar una mayor proporción de pedidos con reseñas negativas dentro del grupo priorizado.

### Umbral de riesgo

El modelo final utiliza un umbral de **0.260125 (26.01%)**, calculado a partir del percentil 90 de sus probabilidades históricas.

Este valor permite distinguir los pedidos clasificados como de riesgo alto.

No significa que el 26.01% de los pedidos recibirá una reseña negativa.

## 4. ¿Cuáles son los límites conocidos del modelo?

Aunque el modelo obtuvo mejores resultados que las alternativas evaluadas, presenta limitaciones importantes.

**Disponibilidad de información:** utiliza datos de entrega real. Por ello, no sirve para anticipar el riesgo antes de entregar el pedido.

**Cambios en las operaciones:** si cambian los transportistas, los tiempos de entrega, los productos o el comportamiento de los clientes, el desempeño podría disminuir.

**Datos históricos:** el modelo fue evaluado con información histórica de Olist. Sus resultados podrían ser diferentes en otros períodos o empresas.

**Predicciones imperfectas:** una probabilidad alta no significa que necesariamente habrá una reseña negativa. Del mismo modo, un pedido con riesgo bajo puede terminar con una mala calificación.

**Alcance de la validación:** los resultados corresponden a la validación interna del proyecto, no a una evaluación con los pedidos ocultos del docente.

El modelo tampoco identifica automáticamente la causa exacta de la insatisfacción; para comprenderla puede ser necesario revisar el pedido y los comentarios del cliente.

## 5. ¿A quién afecta un error del modelo?

El modelo puede cometer dos tipos principales de errores.

### Falso positivo: se clasifica como riesgoso un pedido que no termina con una reseña negativa

**Ejemplo:** un pedido se entrega con retraso, pero el cliente queda satisfecho. El modelo lo considera de riesgo alto y el equipo decide contactarlo.

**¿A quién afecta?**

- **Al equipo de atención:** dedica tiempo a un caso que quizá no requería seguimiento.
- **A la empresa:** utiliza recursos que podrían destinarse a un cliente realmente insatisfecho.
- **Al cliente:** podría recibir una comunicación innecesaria.

### Falso negativo: se clasifica como no riesgoso un pedido que termina con una reseña negativa

**Ejemplo:** un pedido llega a tiempo, pero el producto presenta un problema de calidad. El modelo estima un riesgo bajo y el equipo no lo prioriza.

**¿A quién afecta?**

- **Al cliente:** podría no recibir seguimiento oportuno ante una experiencia negativa.
- **Al vendedor:** podría acumular reclamos o reseñas negativas sin que se identifique el problema a tiempo.
- **A la empresa:** podría perder oportunidades de mejorar el servicio y prevenir insatisfacción recurrente.

**Medida de protección:** las predicciones deben servir para orientar el seguimiento, no para sancionar automáticamente a vendedores ni para negar atención a clientes clasificados como de bajo riesgo.

## 6. ¿Recomendamos utilizar el modelo? ¿Con qué supervisión y cada cuánto revisarlo?

**Sí recomendamos utilizar XGBoost como herramienta de apoyo para priorizar el seguimiento de pedidos entregados**, porque obtuvo mejores resultados que la selección aleatoria, la regla del retraso y la regresión logística en la validación temporal estricta.

Sin embargo, recomendamos una implementación supervisada.

### Supervisión humana

El equipo de atención debe revisar los pedidos priorizados antes de decidir qué acción tomar.

También debe mantener canales normales de atención para clientes que no fueron clasificados como de alto riesgo.

Las decisiones que afecten directamente a vendedores o clientes no deben automatizarse únicamente a partir de la probabilidad del modelo.

### Frecuencia de revisión recomendada

Como propuesta operativa:

- **Mensualmente:** revisar la proporción de reseñas negativas, la precisión de los pedidos priorizados y los casos clasificados incorrectamente.
- **Cada tres meses:** realizar una evaluación más completa para determinar si el modelo y su umbral siguen siendo adecuados.
- **Antes de ese plazo:** revisar el desempeño si ocurren cambios importantes en las entregas, los productos o el comportamiento de los clientes.

Estas frecuencias son recomendaciones de seguimiento, no resultados calculados por el notebook.

### Decisión final

**Recomendamos utilizar el modelo de forma supervisada para apoyar la priorización posterior a la entrega, pero no como sistema de decisión automática.**

Su principal beneficio es ayudar a un equipo con recursos limitados a concentrar sus esfuerzos en los pedidos con mayor probabilidad de insatisfacción.

El modelo debe complementarse con la revisión de cada caso, el análisis de comentarios y el monitoreo periódico de sus resultados.

---

**Resultado principal del proyecto:** XGBoost — Precisión Top 10% de **49.22%** en validación temporal estricta.
