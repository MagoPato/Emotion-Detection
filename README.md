 <h1>Emotion Detection</h1>

  <p>Este proyecto implementa un sistema de detección de emociones en tiempo real utilizando una cámara web. Utiliza una combinación de redes neuronales convolucionales (CNN) para la detección de rostros y un modelo pre-entrenado para la clasificación de emociones.</p>

  <h2>Cómo funciona</h2>

  <p>El sistema consta de dos partes principales:</p>

  <ol>
        <li><strong>Detección de Rostros:</strong> Utiliza un modelo de detección de rostros basado en el método Single Shot Multibox Detector (SSD) para identificar y delimitar los rostros en el video proporcionado por la cámara web.</li>
        <li><strong>Clasificación de Emociones:</strong> Una vez que se detectan los rostros, se extraen y clasifican las emociones asociadas utilizando un modelo de aprendizaje profundo pre-entrenado. El modelo puede reconocer emociones como feliz, triste, enojado, sorprendido, etc.</li>
    </ol>

  <p>El sistema funciona de la siguiente manera:</p>

   <ul>
        <li>Cuando se ejecuta la aplicación Flask (<code>app.py</code>), se inicia un servidor web en el puerto <code>8080</code>.</li>
        <li>Al acceder a la dirección <code>http://127.0.0.1:8080/</code> en un navegador web, se muestra la interfaz de usuario.</li>
        <li>La interfaz de usuario incluye una vista en vivo de la cámara web y un área donde se muestra la emoción detectada.</li>
        <li>Es necesario tener una cámara web habilitada para que la aplicación funcione correctamente.</li>
        <li>Utilizando JavaScript (<code>static/main.js</code>), el video de la cámara web se captura continuamente y se envía al servidor en intervalos regulares.</li>
        <li>En el servidor, las imágenes recibidas se procesan para detectar rostros utilizando un modelo de detección de rostros pre-entrenado. Luego, se extraen los rostros detectados y se clasifican las emociones asociadas utilizando un modelo de clasificación de emociones.</li>
        <li>La emoción detectada se muestra en la interfaz de usuario en tiempo real.</li>
    </ul>

   <h2>Requisitos</h2>

   <ul>
        <li>Python 3.7 o superior</li>
        <li>Una cámara web habilitada</li>
    </ul>

   <h2>Instalación</h2>

  <ol>
        <li>Clona este repositorio en tu máquina local:</li>
        <code>git clone https://github.com/tu_usuario/emotion-detection.git</code>
        <li>Instala las dependencias necesarias. Asegúrate de tener Python 3 y pip instalados. Luego, ejecuta:</li>
        <code>pip install flask tensorflow opencv-python-headless numpy</code>
  </ol>

   <h2>Estructura del proyecto</h2>

   <ul>
      <li><code>app.py</code>: Contiene el servidor Flask y las rutas para la página principal y la predicción de emociones.</li>
        <li><code>modelFEC.h5</code>: El modelo pre-entrenado para la clasificación de emociones.</li>
        <li><code>face_detector/deploy.prototxt</code> y <code>face_detector/res10_300x300_ssd_iter_140000.caffemodel</code>: Modelos de detección de rostros utilizando el método Single Shot Multibox Detector (SSD).</li>
        <li><code>templates/index.html</code>: La plantilla HTML que muestra la interfaz web.</li>
        <li><code>static/main.js</code>: El script JavaScript que maneja la captura de vídeo y el envío de imágenes al servidor.</li>
        <li><code>static/style.css</code>: Estilos CSS para la interfaz web.</li>
    </ul>

   <h2>Contribuyendo</h2>

   <p>¡Las contribuciones son bienvenidas! Si tienes alguna idea para mejorar este proyecto, siéntete libre de abrir un problema o enviar un pull request.</p>

  <h2>Licencia</h2>

   <p>Este proyecto está bajo la licencia MIT. Consulta el archivo <code>LICENSE</code> para más detalles.</p>
