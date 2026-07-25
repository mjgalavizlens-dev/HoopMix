import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tempfile

# 1. Configuración de la página
st.set_page_config(page_title="HoopsAI - Análisis", page_icon="🏀")
st.title("🏀 HoopsAI: Análisis Biomecánico")

# 2. Selector de Modo en la barra lateral
st.sidebar.header("Configuración")
modo_analisis = st.sidebar.radio(
    "¿Qué quieres analizar en este vídeo?",
    ["Solo Mecánica de Tiro", "Solo Salto Vertical", "Tiro en Suspensión (Ambos)"]
)

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Función para calcular ángulos
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radianes = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360.0 - angulo
    return angulo

# 3. Subida de archivo
video_file = st.file_uploader("Sube tu vídeo (mp4, mov)", type=["mp4", "mov"])

if video_file is not None:
    # Guardar video temporalmente
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    st.text("Procesando vídeo... esto puede tardar unos segundos.")
    frame_placeholder = st.empty()

    # Variables de métricas
    min_angulo_codo = 180.0  # Para el Set Point
    max_angulo_codo = 0.0    # Para el Follow-through
    y_tobillo_mas_bajo = 0.0 # Para calcular el despegue
    y_tobillo_mas_alto = 1.0 # (En OpenCV, 0 es arriba, 1 es abajo en coordenadas normalizadas)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Redimensionar para procesar más rápido
            frame = cv2.resize(frame, (640, int(frame.shape[0] * (640 / frame.shape[1]))))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- ANÁLISIS DE TIRO ---
                if modo_analisis in ["Solo Mecánica de Tiro", "Tiro en Suspensión (Ambos)"]:
                    hombro = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    codo = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    muneca = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    angulo_codo = calcular_angulo(hombro, codo, muneca)
                    min_angulo_codo = min(min_angulo_codo, angulo_codo)
                    max_angulo_codo = max(max_angulo_codo, angulo_codo)

                # --- ANÁLISIS DE SALTO ---
                if modo_analisis in ["Solo Salto Vertical", "Tiro en Suspensión (Ambos)"]:
                    tobillo_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    y_tobillo_mas_bajo = max(y_tobillo_mas_bajo, tobillo_y)
                    y_tobillo_mas_alto = min(y_tobillo_mas_alto, tobillo_y)

                # Dibujar esqueleto
                mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Mostrar frame
            frame_placeholder.image(image_rgb, channels="RGB")

    cap.release()

    # 4. Mostrar Resultados Realistas
    st.success("Análisis completado.")

    if modo_analisis in ["Solo Mecánica de Tiro", "Tiro en Suspensión (Ambos)"]:
        st.subheader("🎯 Diagnóstico de Tiro (Brazo Derecho)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Punto de Carga (Mínimo)", value=f"{int(min_angulo_codo)}°")
            if min_angulo_codo < 70:
                st.warning("Estás cerrando demasiado el brazo (encogido). Busca acercarte a los 90° para mayor fluidez y rapidez.")
            elif 75 <= min_angulo_codo <= 100:
                st.info("Buen ángulo de carga. Tienes un Set Point equilibrado.")
            else:
                st.warning("Ángulo de carga muy abierto, puedes perder fuerza desde las piernas.")

        with col2:
            st.metric(label="Extensión (Follow-through)", value=f"{int(max_angulo_codo)}°")
            if max_angulo_codo >= 155:
                st.info("Excelente extensión al soltar el balón. Esto favorece una parábola limpia.")
            else:
                st.warning("Falta extensión completa al soltar (tiro encogido). Trata de estirar el codo por completo apuntando al aro.")

    if modo_analisis in ["Solo Salto Vertical", "Tiro en Suspensión (Ambos)"]:
        st.subheader("🚀 Diagnóstico de Salto")
        diferencia_y = y_tobillo_mas_bajo - y_tobillo_mas_alto
        
        if diferencia_y > 0.05:
            altura_estimada_cm = int(diferencia_y * 150)
            st.metric(label="Altura relativa estimada", value=f"~{altura_estimada_cm} cm")
            if altura_estimada_cm > 40:
                st.info("Buena elevación detectada en la fase de vuelo.")
            else:
                st.warning("Elevación baja. Para tiros en suspensión, asegúrate de transferir la fuerza desde las punteras.")
        else:
            st.error("No se ha detectado un salto claro en el vídeo. Si es un tiro libre, selecciona 'Solo Mecánica de Tiro' en el menú.")
