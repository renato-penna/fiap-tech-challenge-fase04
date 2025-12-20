import cv2
from deepface import DeepFace
from ultralytics import YOLO
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
import json

def detect_activity(keypoints):
    """Detecta atividade baseada nos keypoints"""
    if len(keypoints) < 17:
        return "unknown"
    
    # Keypoints COCO: 0-nose, 5-left_shoulder, 6-right_shoulder, 11-left_hip, 12-right_hip
    # 13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle
    
    shoulders = keypoints[5:7]
    hips = keypoints[11:13]
    knees = keypoints[13:15]
    ankles = keypoints[15:17]
    
    # Verificar se keypoints principais estão visíveis
    if not all(k[0] > 0 and k[1] > 0 for k in [shoulders[0], shoulders[1], hips[0], hips[1]]):
        return "unknown"
    
    # Calcular posições médias
    shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2
    shoulder_x = (shoulders[0][0] + shoulders[1][0]) / 2
    hip_y = (hips[0][1] + hips[1][1]) / 2
    hip_x = (hips[0][0] + hips[1][0]) / 2
    
    # Distância vertical entre ombros e quadris (altura do torso)
    torso_height = abs(shoulder_y - hip_y)
    
    # Alinhamento horizontal (corpo ereto vs inclinado)
    horizontal_offset = abs(shoulder_x - hip_x)
    
    # Diferença de altura entre ombros (detectar deitado)
    shoulder_tilt = abs(shoulders[0][1] - shoulders[1][1])
    
    # Verificar joelhos e tornozelos
    knees_visible = knees[0][0] > 0 and knees[1][0] > 0
    ankles_visible = ankles[0][0] > 0 and ankles[1][0] > 0
    
    if knees_visible:
        knee_y = (knees[0][1] + knees[1][1]) / 2
        hip_to_knee = knee_y - hip_y  # Positivo = joelhos abaixo
    
    if ankles_visible:
        ankle_y = (ankles[0][1] + ankles[1][1]) / 2
    
    # CRITÉRIO 1: EM PÉ (prioridade alta)
    # Torso alto = pessoa ereta
    if torso_height > 120:
        # Se torso muito alto, sempre em pé
        if torso_height > 150:
            return "standing"
        # Torso médio-alto: verificar se não está sentado
        if knees_visible:
            # Joelhos bem abaixo = em pé
            if hip_to_knee > 70:
                return "standing"
            # Joelhos próximos + torso não tão alto = sentado
            if hip_to_knee < 50 and torso_height < 140:
                return "sitting"
        # Sem joelhos visíveis mas torso alto = em pé
        return "standing"
    
    # CRITÉRIO 2: SENTADO
    # Torso curto/médio (tronco comprimido)
    if torso_height < 120:
        # Torso muito curto = sentado
        if torso_height < 80:
            return "sitting"
        # Torso médio: verificar joelhos
        if knees_visible:
            # Joelhos próximos = sentado
            if hip_to_knee < 80:
                return "sitting"
        # Sem joelhos, torso médio-baixo = sentado
        return "sitting"
    
    # CRITÉRIO 3: DEITADO (última prioridade, muito restritivo)
    # Apenas se ombros MUITO inclinados E torso muito curto E corpo horizontal
    if shoulder_tilt > 60 and torso_height < 60 and horizontal_offset < 40:
        return "lying_down"
    
    # Fallback
    return "standing"

def detect_scene_change(frame1, frame2, threshold=0.3):
    """Detecta mudança de cena usando diferença de histograma"""
    if frame1 is None or frame2 is None:
        return False
    
    # Calcular histogramas para cada canal
    hist1 = [cv2.calcHist([frame1], [i], None, [256], [0, 256]) for i in range(3)]
    hist2 = [cv2.calcHist([frame2], [i], None, [256], [0, 256]) for i in range(3)]
    
    # Normalizar histogramas
    hist1 = [cv2.normalize(h, h).flatten() for h in hist1]
    hist2 = [cv2.normalize(h, h).flatten() for h in hist2]
    
    # Calcular correlação para cada canal
    correlations = [cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) for h1, h2 in zip(hist1, hist2)]
    
    # Média das correlações
    avg_correlation = np.mean(correlations)
    
    # Se correlação baixa, é mudança de cena
    return avg_correlation < (1 - threshold)

def combined_detection(video_path, output_path):
    # Inicializar YOLOv11-pose
    yolo_pose = YOLO('yolo11n-pose.pt')
    
    # Estatísticas
    emotion_counter = Counter()
    activity_counter = Counter()
    emotion_occurrences = Counter()  # Conta ocorrências (vezes)
    activity_occurrences = Counter()  # Conta ocorrências (vezes)
    frame_emotions = []
    frame_activities = []
    anomaly_count = 0
    scene_changes = 0
    previous_frame = None
    previous_keypoints = {}
    previous_activities = {}  # Rastrear atividade anterior por pessoa
    previous_emotions = {}  # Rastrear emoção anterior por pessoa
    activity_stability = {}  # Contador de estabilidade para atividades
    emotion_stability = {}  # Contador de estabilidade para emoções
    anomaly_threshold = 100  # Limiar para movimento brusco
    stability_threshold = 15  # Frames necessários para confirmar mudança (0.5s a 30fps)
    scene_change_threshold = 0.4  # Threshold para detecção de cena
    
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
    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar mudança de cena
        is_scene_change = False
        if previous_frame is not None:
            is_scene_change = detect_scene_change(previous_frame, frame, scene_change_threshold)
            if is_scene_change:
                scene_changes += 1
                # Resetar rastreamento em mudança de cena
                previous_keypoints.clear()
                previous_activities.clear()
                previous_emotions.clear()
                activity_stability.clear()
                emotion_stability.clear()
        
        previous_frame = frame.copy()
        frame_emotion_list = []
        frame_activity_list = []

        # Detecção de pose com YOLOv11
        yolo_results = yolo_pose(frame, conf=0.3, verbose=False)
        
        for person_idx, result in enumerate(yolo_results):
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                for person_kpts in result.keypoints.xy:
                    kpts = person_kpts.cpu().numpy()
                    
                    # Detectar anomalias (movimentos bruscos)
                    is_anomaly = False
                    if person_idx in previous_keypoints:
                        prev_kpts = previous_keypoints[person_idx]
                        # Calcular deslocamento médio dos keypoints
                        valid_movements = []
                        for i, (curr, prev) in enumerate(zip(kpts, prev_kpts)):
                            if curr[0] > 0 and curr[1] > 0 and prev[0] > 0 and prev[1] > 0:
                                movement = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                                valid_movements.append(movement)
                        
                        if valid_movements:
                            avg_movement = np.mean(valid_movements)
                            max_movement = np.max(valid_movements)
                            
                            # Detectar movimento anômalo (muito brusco ou atípico)
                            # Ignorar se for mudança de cena
                            if not is_scene_change:
                                if max_movement > anomaly_threshold or avg_movement > anomaly_threshold / 2:
                                    is_anomaly = True
                                    anomaly_count += 1
                    
                    # Armazenar keypoints atuais
                    previous_keypoints[person_idx] = kpts.copy()
                    
                    # Detectar atividade
                    activity = detect_activity(kpts)
                    activity_counter[activity] += 1
                    frame_activity_list.append(activity)
                    
                    # Sistema de estabilização para contar ocorrências
                    if person_idx not in previous_activities:
                        previous_activities[person_idx] = activity
                        activity_stability[person_idx] = 0
                        activity_occurrences[activity] += 1
                    elif previous_activities[person_idx] != activity:
                        # Atividade mudou, começar contador de estabilidade
                        activity_stability[person_idx] += 1
                        if activity_stability[person_idx] >= stability_threshold:
                            # Mudança confirmada após frames estáveis
                            activity_occurrences[activity] += 1
                            previous_activities[person_idx] = activity
                            activity_stability[person_idx] = 0
                    else:
                        # Atividade mantida, resetar contador
                        activity_stability[person_idx] = 0
                    
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
                    
                    # Mostrar atividade no frame
                    if len(kpts) > 0:
                        nose = kpts[0]
                        if nose[0] > 0 and nose[1] > 0:
                            if activity != "unknown":
                                cv2.putText(frame, activity, (int(nose[0]), int(nose[1])-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Detecção de rostos e emoções com RetinaFace
        try:
            faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
            
            for face_obj in faces:
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Desenhar retângulo roxo
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                
                # Analisar emoção
                try:
                    emotion_result = DeepFace.analyze(face_obj['face']*255, actions=['emotion'], enforce_detection=False, detector_backend='skip')
                    if isinstance(emotion_result, list):
                        emotion_result = emotion_result[0]
                    dominant_emotion = emotion_result['dominant_emotion']
                    emotion_counter[dominant_emotion] += 1
                    frame_emotion_list.append(dominant_emotion)
                    
                    # Sistema de estabilização para contar ocorrências de emoção
                    face_id = f"{x//50}_{y//50}"  # ID baseado em região (menos sensível)
                    if face_id not in previous_emotions:
                        previous_emotions[face_id] = dominant_emotion
                        emotion_stability[face_id] = 0
                        emotion_occurrences[dominant_emotion] += 1
                    elif previous_emotions[face_id] != dominant_emotion:
                        # Emoção mudou, começar contador de estabilidade
                        emotion_stability[face_id] += 1
                        if emotion_stability[face_id] >= stability_threshold:
                            # Mudança confirmada após frames estáveis
                            emotion_occurrences[dominant_emotion] += 1
                            previous_emotions[face_id] = dominant_emotion
                            emotion_stability[face_id] = 0
                    else:
                        # Emoção mantida, resetar contador
                        emotion_stability[face_id] = 0
                    
                    # Posicionar label: acima se bbox baixo, ao lado se bbox alto
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Se altura do bbox ocupa mais de 60% da altura da imagem, colocar ao lado
                    if h > (frame_height * 0.6):
                        # Colocar ao lado direito
                        text_x = x + w + 10
                        text_y = y + h // 2
                    else:
                        # Colocar acima (padrão)
                        text_x = x
                        text_y = y - 10
                    
                    cv2.putText(frame, dominant_emotion, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except:
                    pass
        except:
            pass

        frame_emotions.append(frame_emotion_list)
        frame_activities.append(frame_activity_list)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Gerar resumo
    generate_summary(video_path, output_path, emotion_counter, activity_counter, 
                    emotion_occurrences, activity_occurrences, anomaly_count, scene_changes, total_frames, fps)

def generate_summary(video_path, output_path, emotion_counter, activity_counter, 
                    emotion_occurrences, activity_occurrences, anomaly_count, scene_changes, total_frames, fps):
    """Gera resumo das atividades e emoções detectadas"""
    duration = total_frames / fps
    
    summary = {
        "video_info": {
            "input": os.path.basename(video_path),
            "output": os.path.basename(output_path),
            "duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": fps
        },
        "emotions": {
            "by_frames": dict(emotion_counter.most_common()),
            "by_occurrences": dict(emotion_occurrences.most_common())
        },
        "activities": {
            "by_frames": dict(activity_counter.most_common()),
            "by_occurrences": dict(activity_occurrences.most_common())
        },
        "scenes": {
            "total_scene_changes": scene_changes,
            "average_scene_duration": round(duration / (scene_changes + 1), 2) if scene_changes > 0 else duration
        },
        "anomalies": {
            "total_anomalies_detected": anomaly_count,
            "anomaly_rate_per_minute": round((anomaly_count / duration) * 60, 2) if duration > 0 else 0
        },
        "summary": {
            "most_common_emotion": emotion_occurrences.most_common(1)[0] if emotion_occurrences else ("none", 0),
            "most_common_activity": activity_occurrences.most_common(1)[0] if activity_occurrences else ("none", 0),
            "total_emotion_occurrences": sum(emotion_occurrences.values()),
            "total_activity_occurrences": sum(activity_occurrences.values()),
            "total_anomalies": anomaly_count
        }
    }
    
    # Salvar resumo em JSON
    summary_path = output_path.replace('.mp4', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # Imprimir resumo
    print("\n" + "="*60)
    print("RESUMO DO VÍDEO")
    print("="*60)
    print(f"Duração: {duration:.2f}s ({total_frames} frames)")
    print(f"\nEMOÇÕES DETECTADAS:")
    for emotion, count in emotion_occurrences.most_common():
        frames = emotion_counter[emotion]
        percentage = (count / sum(emotion_occurrences.values())) * 100 if sum(emotion_occurrences.values()) > 0 else 0
        print(f"  {emotion}: {count} vezes ({frames} frames, {percentage:.1f}%)")
    
    print(f"\nATIVIDADES DETECTADAS:")
    for activity, count in activity_occurrences.most_common():
        frames = activity_counter[activity]
        percentage = (count / sum(activity_occurrences.values())) * 100 if sum(activity_occurrences.values()) > 0 else 0
        print(f"  {activity}: {count} vezes ({frames} frames, {percentage:.1f}%)")
    
    print(f"\nCENAS DETECTADAS:")
    print(f"  Total de mudanças de cena: {scene_changes}")
    avg_scene_duration = duration / (scene_changes + 1) if scene_changes > 0 else duration
    print(f"  Duração média por cena: {avg_scene_duration:.2f}s")
    
    print(f"\nANOMALIAS DETECTADAS:")
    print(f"  Total de anomalias: {anomaly_count} vezes")
    anomaly_rate = (anomaly_count / duration) * 60 if duration > 0 else 0
    print(f"  Taxa de anomalias: {anomaly_rate:.2f} vezes por minuto")
    
    print(f"\nResumo salvo em: {summary_path}")
    print("="*60)

# Executar
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video/video.mp4')
output_video_path = os.path.join(script_dir, 'video/video_detection_with_report.mp4')

combined_detection(input_video_path, output_video_path)
