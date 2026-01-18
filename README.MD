# 🎬 Transcritor de Vídeos e Imagens com IA

Sistema completo de transcrição de vídeos e extração de texto de imagens (OCR) com análise inteligente usando Gemini AI.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Funcionalidades

### 🎥 Transcrição de Vídeos

- **Múltiplos formatos**: MP4, AVI, MOV, MKV, FLV, WMV
- **Transcrição automática** com Whisper AI (faster-whisper)
- **Detecção automática de idioma**
- **Análise com Gemini AI** usando prompts personalizáveis
- **Estatísticas detalhadas** de processamento

### 🖼️ OCR de Imagens

- **Extração de texto** com Tesseract OCR
- **Suporte a múltiplas imagens** em lote
- **Formatos**: PNG, JPG, JPEG, WEBP, BMP, TIFF
- **OCR em português** otimizado
- **Geração de legendas** inteligentes com Gemini AI

### 📊 Interface Web Moderna

- **Design responsivo** e intuitivo
- **Upload drag & drop** de arquivos
- **Processamento em tempo real** com barra de progresso
- **Estatísticas detalhadas**:
  - Tempo de extração de áudio
  - Tempo de transcrição/OCR
  - Tempo de análise Gemini
  - Caracteres processados
  - Tempo total estimado
- **Visualização e download** de resultados
- **Acesso via rede local**

## 🚀 Instalação

### Pré-requisitos

- Docker e Docker Compose instalados
- Chave da API do Google Gemini ([obtenha aqui](https://aistudio.google.com/))

### Passo a Passo

1. **Clone o repositório**

```bash
git clone https://github.com/seu-usuario/transcritor-ia.git
cd transcritor-ia
```

2. **Configure a API do Gemini**

```bash
# Crie o arquivo .env
echo "GEMINI_API_KEY=sua_chave_aqui" > .env
```

3. **Execute com Docker**

```bash
docker compose up --build
```

4. **Acesse a interface**

```
http://localhost:5000
```

### 🌐 Acesso em Rede Local

Para acessar de outros dispositivos na mesma rede:

1. **Descubra seu IP local**

```bash
hostname -I
# ou
ip addr show
```

2. **Libere a porta no firewall**

```bash
sudo ufw allow 5000/tcp
```

3. **Acesse de qualquer dispositivo**

```
http://SEU_IP_LOCAL:5000
```

Exemplo: `http://192.168.0.77:5000`

## 📖 Como Usar

### Transcrever Vídeos

1. Clique na aba **"🎬 Vídeos"**
2. Faça upload dos vídeos (arraste ou clique)
3. Personalize o prompt para análise do Gemini (opcional)
4. Clique em **"🚀 Processar Vídeos"**
5. Acompanhe o progresso em tempo real
6. Visualize ou baixe os resultados

### Extrair Texto de Imagens

1. Clique na aba **"🖼️ Imagens (OCR)"**
2. Faça upload das imagens com texto
3. Personalize o prompt (ex: "Gere legendas descritivas")
4. Clique em **"🚀 Processar Imagens"**
5. Veja os textos extraídos e análise do Gemini

## 🛠️ Tecnologias

### Backend

- **Python 3.12** - Linguagem principal
- **Flask** - Framework web
- **faster-whisper** - Transcrição de áudio otimizada
- **Tesseract OCR** - Extração de texto de imagens
- **Google Gemini AI** - Análise inteligente
- **FFmpeg** - Processamento de vídeo/áudio
- **Pillow** - Processamento de imagens

### Frontend

- **HTML5/CSS3** - Interface moderna
- **JavaScript (Vanilla)** - Interatividade
- **Design Responsivo** - Funciona em mobile e desktop

### Infraestrutura

- **Docker** - Containerização
- **Docker Compose** - Orquestração

## 📁 Estrutura do Projeto

```
transcritor-ia/
├── app.py                 # Aplicação Flask principal
├── templates/
│   └── index.html        # Interface web
├── Dockerfile            # Container Docker
├── docker-compose.yml    # Orquestração
├── requirements.txt      # Dependências Python
├── .env                  # Variáveis de ambiente (criar)
├── videos/               # Upload de vídeos (auto-criado)
├── imagens/              # Upload de imagens (auto-criado)
├── resultados/           # Resultados processados (auto-criado)
└── README.md
```

## ⚙️ Configuração Avançada

### Variáveis de Ambiente

```bash
# .env
GEMINI_API_KEY=sua_chave_aqui
```

### Portas e Limites

Edite `docker-compose.yml` para customizar:

- **Porta**: Altere `5000:5000` para outra porta
- **Limite de upload**: Configurado em 500MB por padrão

### Modelos Whisper

No `app.py`, você pode alterar o modelo Whisper:

```python
# Opções: tiny, base, small, medium, large
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
```

## 📊 Estatísticas de Processamento

O sistema fornece métricas detalhadas:

- ⏱️ **Tempo de extração de áudio**
- 📝 **Tempo de transcrição/OCR**
- 🤖 **Tempo de análise Gemini**
- 💾 **Tamanho de áudio processado**
- 📄 **Caracteres transcritos**
- ⏰ **Tempo total e estimativa restante**

## 🎯 Casos de Uso

- 📹 **Legendagem automática** de vídeos
- 📚 **Transcrição de aulas** e palestras
- 🎤 **Transcrição de podcasts**
- 📸 **Digitalização de documentos** físicos
- 🖼️ **Extração de texto** de screenshots
- 📊 **Análise de conteúdo** em massa
- 🌐 **Criação de legendas** para redes sociais

## 🔧 Troubleshooting

### Container não inicia

```bash
# Verifique os logs
docker compose logs -f

# Reconstrua o container
docker compose down
docker compose up --build
```

### Erro de API Key

```bash
# Verifique se o .env está correto
cat .env

# Teste a chave em https://aistudio.google.com/
```

### Porta já em uso

```bash
# Altere a porta no docker-compose.yml
ports:
  - "8080:5000"  # Usar porta 8080 ao invés de 5000
```

### Processamento muito lento

- Use um modelo Whisper menor (base ou small)
- Reduza a qualidade do vídeo antes de processar
- Considere usar GPU ao invés de CPU

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- [OpenAI Whisper](https://github.com/openai/whisper) - Modelo de transcrição
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Implementação otimizada
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Engine OCR
- [Google Gemini](https://deepmind.google/technologies/gemini/) - Análise com IA
- [FFmpeg](https://ffmpeg.org/) - Processamento de mídia

## 📧 Contato

Para dúvidas ou sugestões, abra uma [issue](https://github.com/seu-usuario/transcritor-ia/issues).

---

⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!

**Desenvolvido com ❤️ usando Python, IA e Docker**
