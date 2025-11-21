# Sign Language Detector Backend 🤟

A comprehensive backend system for real-time sign language detection and interpretation using state-of-the-art Vision-Language Models (VLMs) and modern web technologies.

## 🎯 Project Overview

This project aims to bridge communication gaps by providing an intelligent system that can accurately detect and interpret sign language gestures in real-time. The system combines computer vision, natural language processing, and web technologies to create an accessible and scalable solution.

### Key Features

- **Real-time Sign Language Detection**: Process video streams and static images
- **Vision-Language Model Integration**: Powered by InternVL3-2B for accurate interpretation
- **Multi-Hardware Support**: Optimized for NVIDIA CUDA, AMD DirectML, and CPU
- **Quantization Support**: 4-bit quantization for efficient inference
- **WebSocket Integration**: Real-time bidirectional communication
- **REST API**: Standard HTTP endpoints for batch processing
- **Fine-tuning Capabilities**: Custom model adaptation for specific sign language dialects

## 🏗️ Architecture

```
sign-language-detector-backend/
├── src/
│   ├── model/                 # VLM Model Components
│   │   ├── model_loader.py    # InternVL3 model loader and inference
│   │   ├── params/            # Model configuration
│   │   │   └── vlm.yml        # VLM parameters and prompts
│   │   └── tests/             # Test data and validation
│   │       └── data/
│   │           ├── images/    # Test images
│   │           └── videos/    # Test videos
│   ├── api/                   # REST API endpoints (planned)
│   ├── websocket/             # WebSocket handlers (planned)
│   └── utils/                 # Utility functions (planned)
├── main.py                    # Application entry point
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## 🚀 Development Stages

### Stage 1: Foundation ✅ (Current)

- [x] Project structure setup
- [x] VLM model integration (InternVL3-2B)
- [x] Multi-hardware support (CUDA/DirectML/CPU)
- [x] Basic video processing capabilities
- [x] Configuration management
- [x] Memory optimization and cleanup

### Stage 2: API Development 🚧 (In Progress)

- [ ] REST API endpoints for image/video processing
- [ ] WebSocket integration for real-time streaming
- [ ] Request/response validation
- [ ] Error handling and logging
- [ ] Authentication and rate limiting

### Stage 3: Model Enhancement 📋 (Planned)

- [ ] Fine-tuning pipeline for custom sign language datasets
- [ ] Performance optimization and benchmarking
- [ ] Model quantization improvements
- [ ] Support for additional VLM architectures
- [ ] Batch processing capabilities

### Stage 4: Production Readiness 📋 (Planned)

- [ ] Docker containerization
- [ ] Monitoring and health checks
- [ ] Comprehensive testing suite
- [ ] Documentation and API references
- [ ] CI/CD pipeline setup

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.11+**: Primary development language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **InternVL3-2B**: Vision-Language Model for sign language interpretation

### Hardware Acceleration

- **CUDA**: NVIDIA GPU acceleration
- **DirectML**: AMD GPU acceleration via PyTorch-DirectML
- **BitsAndBytes**: 4-bit quantization for memory efficiency

### API & Communication

- **FastAPI**: Modern web framework (planned)
- **WebSockets**: Real-time communication (planned)
- **Uvicorn**: ASGI server (planned)

### Development Tools

- **UV**: Fast Python package installer
- **YAML**: Configuration management
- **Python-dotenv**: Environment variable management

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- CUDA-compatible GPU (optional, for acceleration)
- AMD GPU with DirectML support (optional, for acceleration)

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/Okeha/sign-language-detector-backend.git
   cd sign-language-detector-backend
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install uv
   uv sync
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   echo "HF_TOKEN=your_hugging_face_token_here" > .env
   ```

5. **Test the installation**
   ```bash
   python src/model/model_loader.py
   ```

## 🚀 Usage

### Current Functionality

#### Basic Video Processing

```python
from src.model.model_loader import VLMModelLoader

# Initialize the model loader
vlm_loader = VLMModelLoader()

# Process a video file
result = vlm_loader.generate_response("path/to/your/video.mp4")

# Clean up resources
vlm_loader.shutdown()
```

#### Configuration

Modify `src/model/params/vlm.yml` to customize:

- Model selection
- Inference prompts
- Generation parameters

```yaml
model: OpenGVLab/InternVL3-2B-hf
prompt: You are a sign language interpreter. Given the video input, provide a concise and accurate text translation of the sign language being communicated.
```

### Planned API Endpoints

```bash
# Process single image
POST /api/v1/detect/image

# Process video file
POST /api/v1/detect/video

# Real-time WebSocket endpoint
WS /ws/detect/stream

# Health check
GET /api/v1/health

# Model information
GET /api/v1/model/info
```

## 🔧 Hardware Requirements

### Minimum Requirements

- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

### GPU Requirements (Optional but Recommended)

- **NVIDIA GPU**: GTX 1060 / RTX 2060 or better with 6GB+ VRAM
- **AMD GPU**: RX 6600 or better with DirectML support
- **VRAM**: 4GB minimum, 8GB+ recommended

## 🧪 Testing

### Test Data Structure

```
src/model/tests/data/
├── images/          # Test images for static detection
└── videos/          # Test videos for sequence detection
    └── test2.mp4    # Sample test video
```

### Running Tests

```bash
# Test model loading and inference
python src/model/model_loader.py

# Run API tests (when implemented)
python -m pytest tests/

# Performance benchmarking (when implemented)
python scripts/benchmark.py
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
```

## 📊 Performance

### Current Benchmarks

- **Model Loading**: ~15-30 seconds (depending on hardware)
- **Video Processing**: ~2-5 seconds per video (8 frames)
- **Memory Usage**: ~4-8GB VRAM (with 4-bit quantization)

### Optimization Features

- 4-bit quantization reduces memory usage by ~75%
- DirectML support for AMD GPUs
- Efficient memory cleanup and garbage collection
- Configurable generation parameters

## 🔮 Future Enhancements

### Short-term Goals

- REST API implementation with FastAPI
- WebSocket support for real-time processing
- Comprehensive error handling and logging
- Docker containerization

### Long-term Vision

- Support for multiple sign languages (ASL, BSL, etc.)
- Custom fine-tuning interface
- Mobile app integration
- Cloud deployment options
- Real-time video streaming optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenGVLab** for the InternVL3 model
- **Hugging Face** for the Transformers library
- **PyTorch Team** for the deep learning framework
- **Microsoft** for DirectML AMD GPU support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Okeha/sign-language-detector-backend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Okeha/sign-language-detector-backend/discussions)
- **Email**: anthony.okeh@example.com

---

**Made with ❤️ to bridge communication gaps and make technology more accessible.**
