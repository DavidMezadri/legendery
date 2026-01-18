from flask import Flask, render_template, request, jsonify, send_file
from faster_whisper import WhisperModel
import ffmpeg
import os
import google.generativeai as genai
from pathlib import Path
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
from PIL import Image
import pytesseract

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/app/videos'
app.config['IMAGES_FOLDER'] = '/app/imagens'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY não definida. Verifique o .env ou docker-compose")

AUDIO_DIR = "/app/audio_temp"
OUTPUT_DIR = "/app/resultados"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estado global para tracking
processing_status = {
    'is_processing': False,
    'current_video': None,
    'current_step': '',
    'progress': 0,
    'total_videos': 0,
    'processed_videos': 0,
    'message': 'Aguardando...',
    'start_time': None,
    'elapsed_time': 0,
    'estimated_remaining': 0,
    'stats': {
        'audio_extraction_time': 0,
        'transcription_time': 0,
        'gemini_analysis_time': 0,
        'total_audio_size': 0,
        'total_transcription_length': 0
    }
}

# Carrega modelo Whisper
whisper_model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("models/gemini-2.5-flash-lite")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE_EXTENSIONS']

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    return whisper_model

def extrair_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(
        audio_path, ac=1, ar=16000
    ).overwrite_output().run(quiet=True)

def transcrever_audio(audio_path, model):
    segments, info = model.transcribe(audio_path, language="pt")
    transcricao = "\n".join([segment.text.strip() for segment in segments])
    return transcricao, info.language

def analisar_com_gemini(transcricao, prompt_custom):
    if not GEMINI_API_KEY:
        return None
    
    prompt = f"{prompt_custom}\n\nTranscrição:\n{transcricao}"
    try:
        print(f"[INFO] Enviando {len(transcricao)+len(prompt_custom)} caracteres para Gemini...")
        response = model_gemini.generate_content(prompt)
        print(f"[INFO] Resposta recebida do Gemini")
        return response.text
    except Exception as e:
        print(f"[ERRO] Falha na API do Gemini: {str(e)}")
        return f"Erro ao analisar com Gemini: {str(e)}"

def extrair_texto_imagem(image_path):
    """Extrai texto de uma imagem usando Tesseract OCR"""
    try:
        print(f"[INFO] Extraindo texto de: {os.path.basename(image_path)}")
        
        # Abrir imagem
        img = Image.open(image_path)
        
        # Configurar Tesseract para português
        custom_config = r'--oem 3 --psm 6 -l por'
        
        # Extrair texto
        texto = pytesseract.image_to_string(img, config=custom_config)
        
        print(f"[INFO] Texto extraído: {len(texto)} caracteres")
        return texto.strip()
    except Exception as e:
        print(f"[ERRO] Falha ao extrair texto: {str(e)}")
        return f"[ERRO: {str(e)}]"

def processar_imagens_background(image_files, custom_prompt):
    """Processa múltiplas imagens com OCR e envia para Gemini"""
    global processing_status
    import time
    
    processing_status['is_processing'] = True
    processing_status['total_videos'] = len(image_files)
    processing_status['processed_videos'] = 0
    processing_status['start_time'] = time.time()
    processing_status['message'] = 'Iniciando processamento de imagens...'
    
    resultados = []
    textos_extraidos = []
    
    for i, image_file in enumerate(image_files, 1):
        image_name = Path(image_file).stem
        processing_status['current_video'] = image_name
        processing_status['processed_videos'] = i - 1
        processing_status['current_step'] = '🖼️ Processando imagem'
        processing_status['message'] = f'[{i}/{len(image_files)}] Preparando {image_name}...'
        
        try:
            # Extração de texto com OCR
            processing_status['current_step'] = '📝 Extraindo texto (OCR)'
            processing_status['message'] = f'[{i}/{len(image_files)}] Extraindo texto de {image_name}...'
            step_start = time.time()
            
            texto_extraido = extrair_texto_imagem(image_file)
            ocr_time = time.time() - step_start
            
            textos_extraidos.append({
                'imagem': image_name,
                'texto': texto_extraido,
                'ordem': i
            })
            
            processing_status['stats']['transcription_time'] += ocr_time
            processing_status['stats']['total_transcription_length'] += len(texto_extraido)
            
            print(f"[INFO] OCR completo em {ocr_time:.2f}s")
            
            # Atualizar progresso
            processing_status['processed_videos'] = i
            processing_status['progress'] = int((i / len(image_files)) * 100)
            
            # Calcular tempo estimado
            elapsed = time.time() - processing_status['start_time']
            processing_status['elapsed_time'] = int(elapsed)
            if i > 0:
                avg_time_per_image = elapsed / i
                remaining_images = len(image_files) - i
                processing_status['estimated_remaining'] = int(avg_time_per_image * remaining_images)
                
        except Exception as e:
            print(f"[ERRO] Falha ao processar {image_name}: {str(e)}")
            textos_extraidos.append({
                'imagem': image_name,
                'texto': f"[ERRO: {str(e)}]",
                'ordem': i
            })
    
    # Combinar todos os textos
    processing_status['current_step'] = '📋 Combinando textos'
    processing_status['message'] = 'Combinando textos extraídos...'
    
    texto_completo = "\n\n".join([
        f"=== Imagem {t['ordem']}: {t['imagem']} ===\n{t['texto']}"
        for t in textos_extraidos
    ])
    
    # Analisar com Gemini
    processing_status['current_step'] = '🤖 Analisando com Gemini AI'
    processing_status['message'] = 'Enviando textos para Gemini...'
    
    if GEMINI_API_KEY:
        step_start = time.time()
        analise = analisar_com_gemini(texto_completo, custom_prompt)
        gemini_time = time.time() - step_start
        processing_status['stats']['gemini_analysis_time'] = gemini_time
        print(f"[INFO] Análise Gemini completa em {gemini_time:.2f}s")
    else:
        analise = None
        print(f"[AVISO] GEMINI_API_KEY não configurada - pulando análise")
    
    # Salvar resultados
    processing_status['current_step'] = '💾 Salvando resultados'
    processing_status['message'] = 'Salvando resultados...'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    texto_path = os.path.join(OUTPUT_DIR, f"imagens_{timestamp}_textos.txt")
    analise_path = os.path.join(OUTPUT_DIR, f"imagens_{timestamp}_analise.txt")
    resultado_json = os.path.join(OUTPUT_DIR, f"imagens_{timestamp}_resultado.json")
    
    # Salvar textos extraídos
    with open(texto_path, "w", encoding="utf-8") as f:
        f.write(texto_completo)
    
    # Salvar análise
    if analise:
        with open(analise_path, "w", encoding="utf-8") as f:
            f.write(analise)
    
    # Salvar JSON
    resultado = {
        "tipo": "imagens",
        "total_imagens": len(image_files),
        "timestamp": datetime.now().isoformat(),
        "textos_extraidos": textos_extraidos,
        "texto_completo": texto_completo,
        "analise_gemini": analise,
        "stats": {
            "ocr_time": round(processing_status['stats']['transcription_time'], 2),
            "gemini_analysis_time": round(processing_status['stats']['gemini_analysis_time'], 2),
            "total_chars": len(texto_completo)
        }
    }
    
    with open(resultado_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    processing_status['is_processing'] = False
    processing_status['current_step'] = '✅ Concluído'
    processing_status['message'] = f'Processamento finalizado! {len(image_files)} imagens processadas.'
    processing_status['progress'] = 100
    print(f"[INFO] Processamento de imagens completo! Total: {len(image_files)} imagens")

def processar_videos_background(video_files, custom_prompt):
    global processing_status
    import time
    
    processing_status['is_processing'] = True
    processing_status['total_videos'] = len(video_files)
    processing_status['processed_videos'] = 0
    processing_status['start_time'] = time.time()
    processing_status['message'] = 'Iniciando processamento...'
    
    model = get_whisper_model()
    resultados = []
    
    for i, video_file in enumerate(video_files, 1):
        video_name = Path(video_file).stem
        processing_status['current_video'] = video_name
        processing_status['processed_videos'] = i - 1
        processing_status['current_step'] = '🎬 Preparando vídeo'
        processing_status['message'] = f'[{i}/{len(video_files)}] Preparando {video_name}...'
        
        # Paths
        audio_path = os.path.join(AUDIO_DIR, f"{video_name}.wav")
        transcricao_path = os.path.join(OUTPUT_DIR, f"{video_name}_transcricao.txt")
        analise_path = os.path.join(OUTPUT_DIR, f"{video_name}_analise.txt")
        resultado_json = os.path.join(OUTPUT_DIR, f"{video_name}_resultado.json")
        
        # Processar
        try:
            # Etapa 1: Extração de áudio
            processing_status['current_step'] = '🎵 Extraindo áudio'
            processing_status['message'] = f'[{i}/{len(video_files)}] Extraindo áudio de {video_name}...'
            print(f"[INFO] Extraindo áudio: {video_name}")
            step_start = time.time()
            extrair_audio(video_file, audio_path)
            audio_time = time.time() - step_start
            processing_status['stats']['audio_extraction_time'] += audio_time
            print(f"[INFO] Áudio extraído em {audio_time:.2f}s")
            
            # Tamanho do áudio
            audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
            processing_status['stats']['total_audio_size'] += audio_size
            
            # Etapa 2: Transcrição
            processing_status['current_step'] = '📝 Transcrevendo áudio'
            processing_status['message'] = f'[{i}/{len(video_files)}] Transcrevendo {video_name}... (pode demorar)'
            print(f"[INFO] Iniciando transcrição: {video_name}")
            step_start = time.time()
            transcricao, idioma = transcrever_audio(audio_path, model)
            transcription_time = time.time() - step_start
            processing_status['stats']['transcription_time'] += transcription_time
            processing_status['stats']['total_transcription_length'] += len(transcricao)
            print(f"[INFO] Transcrição completa em {transcription_time:.2f}s ({len(transcricao)} caracteres)")
            
            with open(transcricao_path, "w", encoding="utf-8") as f:
                f.write(transcricao)
            
            # Etapa 3: Análise com Gemini
            processing_status['current_step'] = '🤖 Analisando com Gemini AI'
            processing_status['message'] = f'[{i}/{len(video_files)}] Enviando para Gemini: {video_name}...'
            print(f"[INFO] Enviando para Gemini: {video_name}")
            
            if GEMINI_API_KEY:
                step_start = time.time()
                analise = analisar_com_gemini(transcricao, custom_prompt)
                gemini_time = time.time() - step_start
                processing_status['stats']['gemini_analysis_time'] += gemini_time
                print(f"[INFO] Análise Gemini completa em {gemini_time:.2f}s")
            else:
                analise = None
                gemini_time = 0
                print(f"[AVISO] GEMINI_API_KEY não configurada - pulando análise")
            
            if analise:
                with open(analise_path, "w", encoding="utf-8") as f:
                    f.write(analise)
            
            # Salvar resultado
            processing_status['current_step'] = '💾 Salvando resultados'
            processing_status['message'] = f'[{i}/{len(video_files)}] Salvando resultados de {video_name}...'
            resultado = {
                "video": video_name,
                "idioma": idioma,
                "timestamp": datetime.now().isoformat(),
                "transcricao": transcricao,
                "analise": analise,
                "stats": {
                    "audio_extraction_time": round(audio_time, 2),
                    "transcription_time": round(transcription_time, 2),
                    "gemini_analysis_time": round(gemini_time, 2),
                    "audio_size_mb": round(audio_size, 2),
                    "transcription_chars": len(transcricao)
                }
            }
            
            with open(resultado_json, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)
            
            resultados.append(resultado)
            os.remove(audio_path)
            print(f"[INFO] Vídeo processado com sucesso: {video_name}")
            
            # Atualizar progresso
            processing_status['processed_videos'] = i
            processing_status['progress'] = int((i / len(video_files)) * 100)
            
            # Calcular tempo estimado
            elapsed = time.time() - processing_status['start_time']
            processing_status['elapsed_time'] = int(elapsed)
            if i > 0:
                avg_time_per_video = elapsed / i
                remaining_videos = len(video_files) - i
                processing_status['estimated_remaining'] = int(avg_time_per_video * remaining_videos)
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            resultados.append({
                "video": video_name,
                "erro": str(e)
            })
    
    processing_status['is_processing'] = False
    processing_status['current_step'] = '✅ Concluído'
    processing_status['message'] = f'Processamento finalizado! {len(resultados)} vídeos processados.'
    processing_status['progress'] = 100
    print(f"[INFO] Processamento completo! Total: {len(resultados)} vídeos")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'videos' not in request.files and 'images' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    uploaded_videos = []
    uploaded_images = []
    
    # Upload de vídeos
    if 'videos' in request.files:
        files = request.files.getlist('videos')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_videos.append(filename)
    
    # Upload de imagens
    if 'images' in request.files:
        files = request.files.getlist('images')
        for file in files:
            if file and allowed_image(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['IMAGES_FOLDER'], filename)
                file.save(filepath)
                uploaded_images.append(filename)
    
    return jsonify({
        'success': True,
        'uploaded_videos': uploaded_videos,
        'uploaded_images': uploaded_images,
        'total': len(uploaded_videos) + len(uploaded_images)
    })

@app.route('/process', methods=['POST'])
def process_videos():
    if processing_status['is_processing']:
        return jsonify({'error': 'Já existe um processamento em andamento'}), 400
    
    data = request.json
    custom_prompt = data.get('prompt', 'Resuma o conteúdo desta transcrição.')
    process_type = data.get('type', 'videos')  # 'videos' ou 'images'
    
    if process_type == 'images':
        images = [
            os.path.join(app.config['IMAGES_FOLDER'], f)
            for f in os.listdir(app.config['IMAGES_FOLDER'])
            if allowed_image(f)
        ]
        
        if not images:
            return jsonify({'error': 'Nenhuma imagem encontrada'}), 400
        
        # Processar imagens em background
        thread = threading.Thread(target=processar_imagens_background, args=(images, custom_prompt))
        thread.start()
    else:
        videos = [
            os.path.join(app.config['UPLOAD_FOLDER'], f)
            for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if allowed_file(f)
        ]
        
        if not videos:
            return jsonify({'error': 'Nenhum vídeo encontrado'}), 400
        
        # Processar vídeos em background
        thread = threading.Thread(target=processar_videos_background, args=(videos, custom_prompt))
        thread.start()
    
    return jsonify({'success': True, 'message': 'Processamento iniciado'})

@app.route('/status')
def get_status():
    return jsonify(processing_status)

@app.route('/results')
def list_results():
    results = []
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('_resultado.json'):
            filepath = os.path.join(OUTPUT_DIR, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
    
    return jsonify(results)

@app.route('/download/<filename>')
def download_file(filename):
    # Decodifica o filename caso venha com caracteres especiais
    from urllib.parse import unquote
    filename = unquote(filename)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Debug: lista arquivos disponíveis
    print(f"Tentando baixar: {filename}")
    print(f"Caminho completo: {filepath}")
    print(f"Existe? {os.path.exists(filepath)}")
    print(f"Arquivos disponíveis: {os.listdir(OUTPUT_DIR)}")
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    
    return jsonify({
        'error': 'Arquivo não encontrado',
        'requested': filename,
        'available': os.listdir(OUTPUT_DIR)
    }), 404

@app.route('/videos')
def list_videos():
    videos = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if allowed_file(f)
    ]
    return jsonify(videos)

@app.route('/images')
def list_images():
    images = [
        f for f in os.listdir(app.config['IMAGES_FOLDER'])
        if allowed_image(f)
    ]
    return jsonify(images)

@app.route('/delete-video/<filename>', methods=['DELETE'])
def delete_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'{filename} removido'})
    return jsonify({'error': 'Arquivo não encontrado'}), 404

@app.route('/delete-image/<filename>', methods=['DELETE'])
def delete_image(filename):
    filepath = os.path.join(app.config['IMAGES_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'{filename} removido'})
    return jsonify({'error': 'Arquivo não encontrado'}), 404

@app.route('/clear-videos', methods=['DELETE'])
def clear_all_videos():
    count = 0
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(file):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
            count += 1
    return jsonify({'success': True, 'message': f'{count} vídeo(s) removido(s)'})

@app.route('/clear-images', methods=['DELETE'])
def clear_all_images():
    count = 0
    for file in os.listdir(app.config['IMAGES_FOLDER']):
        if allowed_image(file):
            os.remove(os.path.join(app.config['IMAGES_FOLDER'], file))
            count += 1
    return jsonify({'success': True, 'message': f'{count} imagem(ns) removida(s)'})

@app.route('/clear-results', methods=['DELETE'])
def clear_results():
    count = 0
    for file in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, file))
        count += 1
    return jsonify({'success': True, 'message': f'{count} resultado(s) removido(s)'})


@app.route('/view/<path:filename>', methods=['GET'])
def view_file(filename):
    file_path = os.path.join("/app/resultados", filename)

    # Proteção básica
    if not os.path.isfile(file_path):
        return jsonify({"error": "Arquivo não encontrado"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return jsonify({
            "filename": filename,
            "content": content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)