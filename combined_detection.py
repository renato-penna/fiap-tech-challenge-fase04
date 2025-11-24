import cv2
from deepface import DeepFace
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

def combined_detection(video_path, output_path):
    # Inicializar MediaPipe Pose com parâmetros otimizados
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.3,  # Reduzido para detectar mais
        min_tracking_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Variáveis para otimização
    last_pose_landmarks = None
    frames_without_detection = 0

    # Capturar vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Processar frames
    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Pré-processamento: melhorar contraste e iluminação
        frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        # Equalização de histograma para melhor detecção
        lab = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Detecção de pose com MediaPipe
        rgb_frame = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            last_pose_landmarks = pose_results.pose_landmarks
            frames_without_detection = 0
            
            # Desenhar landmarks com estilo
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        elif last_pose_landmarks is not None and frames_without_detection < 5:
            # Usar última pose detectada se falhar por poucos frames
            frames_without_detection += 1
            mp_drawing.draw_landmarks(
                frame,
                last_pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            frames_without_detection += 1

        # Detecção de rostos e emoções com RetinaFace
        try:
            faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
            
            for face_obj in faces:
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Desenhar retângulo
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Analisar emoção
                try:
                    emotion_result = DeepFace.analyze(face_obj['face']*255, actions=['emotion'], enforce_detection=False, detector_backend='skip')
                    if isinstance(emotion_result, list):
                        emotion_result = emotion_result[0]
                    dominant_emotion = emotion_result['dominant_emotion']
                    cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except:
                    pass
        except:
            pass

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Executar
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video/5patetas.mp4')
output_video_path = os.path.join(script_dir, 'video/5patetas_detection.mp4')

combined_detection(input_video_path, output_video_path)
