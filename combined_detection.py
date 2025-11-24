import cv2
from deepface import DeepFace
from ultralytics import YOLO
import os
import numpy as np
from tqdm import tqdm

def combined_detection(video_path, output_path):
    # Inicializar YOLOv11-pose
    yolo_pose = YOLO('yolo11n-pose.pt')
    
    # Conexões do skeleton (COCO format)
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]

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
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detecção de pose com YOLOv11
        yolo_results = yolo_pose(frame, conf=0.3, verbose=False)
        
        for result in yolo_results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                for person_kpts in result.keypoints.xy:
                    kpts = person_kpts.cpu().numpy()
                    
                    # Desenhar skeleton
                    for connection in skeleton:
                        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
                        if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                            pt1 = kpts[pt1_idx]
                            pt2 = kpts[pt2_idx]
                            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                                        (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
                    
                    # Desenhar keypoints
                    for x, y in kpts:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

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
input_video_path = os.path.join(script_dir, 'video/medico.mp4')
output_video_path = os.path.join(script_dir, 'video/medico_detection.mp4')

combined_detection(input_video_path, output_video_path)
