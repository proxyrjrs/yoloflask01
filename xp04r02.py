###############################################################################################
#                    Versão de Reconhecimento de Objetos usando Yolo5 e Flask
#    Interface minimalista usando templates html, css e js para adicionar funcionalidades
#    - Com Opção de Captura de Vídeo via WebCam
#    - Com Opção de Upload de Vídeo via WebForm
###############################################################################################

import os
import cv2
import torch
import time
import logging
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

#
# Habilitando o Log - Opção de Salvar em Arquivo Desabilitada
#

logging.basicConfig(
    #filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S'
)

#
# Criando objeto de aplicação Flask em app ...
#

app = Flask(__name__)

#
# Preparando diretório para salvar 'uploads' efetuados ...
#

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Pasta '{UPLOAD_FOLDER}' criada.")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#
# Flags Lógicos para Ativar/Desativar Stream/Video ...
#


stop_live = False
stop_video = False

#
# Faz a carga do 'ultralytics/yolov5' - modelo pré-treinado ...
#

logging.info("Carregando o modelo YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
logging.info("Modelo YOLOv5 carregado com sucesso.")

#
# Dicionário para mapear alguns rótulos do yolo5
#

label_mapping = {
    'person': 'pessoa',
    'bicycle': 'bicicleta',
    'car': 'carro',
    'motorcycle': 'motocicleta',
    'airplane': 'avião',
    'bus': 'ônibus',
    'train': 'trem',
    'truck': 'caminhão',
    'boat': 'barco',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'stop sign': 'placa de parada',
    'parking meter': 'parquímetro',
    'bench': 'banco',
    'bird': 'pássaro',
    'cat': 'gato',
    'dog': 'cachorro',
    'horse': 'cavalo',
    'sheep': 'ovelha',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'urso',
    'zebra': 'zebra',
    'giraffe': 'girafa',
    'backpack': 'mochila',
    'umbrella': 'guarda-chuva',
    'handbag': 'bolsa',
    'tie': 'gravata',
    'suitcase': 'mala',
    'frisbee': 'disco',
    'skis': 'esquis',
    'snowboard': 'prancha de snowboard',
    'sports ball': 'bola esportiva',
    'kite': 'pipa',
    'baseball bat': 'taco de beisebol',
    'baseball glove': 'luva de beisebol',
    'skateboard': 'skate',
    'surfboard': 'prancha de surfe',
    'tennis racket': 'raquete de tênis',
    # Outros objetos no CocoDataset Que Precisarem Ser Mapeados ...
}

def process_frame(frame):
    """Aplica a detecção de objetos no frame e desenha as bounding boxes com rótulos em português."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detections = results.xyxy[0]
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label_en = model.names[int(cls)]
        label_pt = label_mapping.get(label_en, label_en)
        label_text = f"{label_pt} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def gen_live_frames():
    """Gera frames da câmera (captura ao vivo) com detecção de objetos."""
    global stop_live
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Erro ao abrir a câmera.")
        return
    logging.info("Câmera aberta para streaming ao vivo.")
    while True:
        if stop_live:
            logging.info("Streaming ao vivo interrompido pelo usuário.")
            break
        ret, frame = cap.read()
        if not ret:
            logging.error("Falha ao capturar frame da câmera.")
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Falha ao codificar o frame.")
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    stop_live = False

def gen_video_frames(video_path):
    """Gera frames a partir de um vídeo enviado com detecção de objetos."""
    global stop_video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Erro ao abrir o vídeo.")
        return
    logging.info("Streaming de vídeo iniciado.")
    while cap.isOpened():
        if stop_video:
            logging.info("Streaming de vídeo interrompido pelo usuário.")
            break
        ret, frame = cap.read()
        if not ret:
            logging.info("Fim do vídeo.")
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)
    cap.release()
    stop_video = False

"""
Configuração de Rotas na Aplicação Flask-Web
"""
@app.route('/stop_live')
def stop_live_route():
    global stop_live
    stop_live = True
    return jsonify({"status": "live streaming stopped"})

@app.route('/stop_video')
def stop_video_route():
    global stop_video
    stop_video = True
    return jsonify({"status": "video streaming stopped"})

# Rotas da aplicação
@app.route('/')
def index():
    return render_template('index02.html')

@app.route('/live_stream')
def live_stream():
    """Página que exibe o streaming ao vivo com interface de controle."""
    return render_template('live_stream02.html')

@app.route('/live_feed')
def live_feed():
    """Retorna o streaming MJPEG do feed ao vivo."""
    return Response(gen_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "Nenhum arquivo enviado", 400
    file = request.files['video']
    if file.filename == '':
        return "Nenhum arquivo selecionado", 400
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    logging.info(f"Vídeo salvo: {video_path}")
    return redirect(url_for('video_stream', filename=filename))

@app.route('/video_stream')
def video_stream():
    filename = request.args.get('filename')
    if not filename:
        return "Arquivo não especificado", 400
    return render_template('video_stream02.html', filename=filename)

@app.route('/video_feed_video/<filename>')
def video_feed_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(gen_video_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


"""
Inicia aplicação e ativa logging na saída padrão (ou arquivo de log)
"""
if __name__ == '__main__':
    logging.info("Iniciando a aplicação Flask.")
    app.run(debug=True)
