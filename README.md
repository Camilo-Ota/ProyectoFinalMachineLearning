## Proyecto Final Aprendizaje de maquina

Andrea Carolina Forero<br>
Camilo Andres Otalora <br>

### Proposito:


El propósito de nuestro trabajo es abordar la temática de BNPL, que significa Compra ahora, paga después. Esta es una funcionalidad que permite a los clientes adquirir productos o servicios al instante y aplazar el pago en varias cuotas. De esta manera, abordaremos un problema sobre predecir si un cliente podría tener incumplimientos en los pagos o no. Mediante modelos de clasificación, como la regresión logística, Random Forest y XGBoost. Además, identificar las variables con patrones e influencias más relevantes sobre la target, con el fin de mejorar la toma de decisiones en las gestiones de crédito. 

### Problema:

Múltiples empresas ofrecen estos servicios de financiamiento tipo BNPL para todo tipo de público, desde personas sin empleo, asalariadas, estudiantes, etc. Aquí enfrentando un alto riesgo asociado al incumplimiento de los pagos por parte de estos clientes.


En la mayoría de los casos, los usuarios adquieren productos a cuotas sin pensar antes en la capacidad real de compra que tienen estos, generando retrasos en los pagos, pagos perdidos e incluso incumplimiento total del cliente.


Esto afectando directamente a:


* La rentabilidad de la empresa

* La sostenibilidad del modelo de negocio

* La correcta asignación de crédito


Por lo tanto, una mala clasificación en los modelos podría generar: aprobar clientes con alto riesgo o rechazar clientes confiables.


### Objetivo general

Desarrollar y evaluar modelos de clasificación que permitan predecir si un cliente incurrirá en incumplimiento de pago (default_flag), optimizando el desempeño del modelo en la identificación de clientes de alto riesgo.


### Objetivos especificos

1) Construir modelos de clasificación como regresión logística y árbol de decisión para la predicción de default_flag( Pago o incumplimiento de pago)
2) Evaluar el desempeño de los modelos utilizando métricas como recall, precisión, F1-score y ROC-AUC.
3) Ajustar los hiperparámetros para mejorar el rendimiento de los modelos.
4) Comparar los modelos desarrollados para identificar cuál presenta mejor desempeño.
5) Identificar las variables más influyentes en la predicción del incumplimiento de pago.


### Variable objetivo en el analisis

La variable objetivo o target de los modelos es "default_flag", que nos va a decir, según las características si:
- 1 -> El cliente incumplió
- 0 -> El cliente pagó correctamente.

Donde es esencial tener en cuenta posibles equivocaciones del modelo, que podrían ser: 




* Falso positivo: El modelo puede predecir que la persona haya incumplido con el pago, pero en realidad sí lo hizo. Afectando que el modelo esté rechazando clientes buenos, perdiendo ingresos y dando una mala experiencia al usuario que quiere utilizar estos servicios.

* Falso negativo: El modelo dice que el cliente pagará, pero en realidad el cliente incumplirá con los pagos. Ocasionando pérdidas económicas directas e incrementando el riesgo financiero.