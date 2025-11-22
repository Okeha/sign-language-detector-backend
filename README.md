# Sign Language Detector Backend 🤟

A comprehensive backend system for real-time sign language detection and interpretation using state-of-the-art Vision-Language Models (VLMs) and modern web technologies.

## 🎯 Project Overview

This project aims to bridge communication gaps by providing an intelligent system that can accurately detect and interpret sign language gestures in real-time. The system combines computer vision, natural language processing, and web technologies to create an accessible and scalable solution.

### Key Features

- **Real-time Sign Language Detection**: Process video streams and static images
- **Vision-Language Model Integration**: Powered by InternVL3-2B for accurate interpretation
- **Multi-Hardware Support**: Optimized for NVIDIA CUDA, AMD DirectML, and CPU
- **Fine-tuning Pipeline**: Complete SFT training with PEFT/LoRA for efficient adaptation
- **Custom Data Collation**: Video frame sampling with proper label masking for causal LM
- **Stratified Dataset Splitting**: Gloss-aware train/test splits ensuring balanced representation
- **Quantization Support**: 4-bit quantization for efficient inference (CUDA only)
- **Automated Video Processing**: Frame extraction and preprocessing with Decord

## 🏗️ Architecture

```
sign-language-detector-backend/
├── src/
│   ├── __init__.py                # Package marker
│   ├── model/                     # VLM Model Components
│   │   ├── __init__.py            # Package marker
│   │   ├── model_loader.py        # InternVL3 model loader and inference
│   │   ├── params/                # Model configuration
│   │   │   └── vlm.yml            # VLM parameters and prompts
│   │   ├── finetune/              # Fine-tuning pipeline (MAIN TRAINING PIPELINE)
│   │   │   ├── __init__.py        # Package marker
│   │   │   ├── data_engineering/  # Dataset processing
│   │   │   │   ├── __init__.py    # Package marker
│   │   │   │   ├── video_downloader.py     # YouTube video downloader
│   │   │   │   ├── filter_data.py          # WLASL data filtering
│   │   │   │   ├── raw_videos/             # Downloaded video files
│   │   │   │   └── datasets/               # Processed datasets (JSON)
│   │   │   ├── dataset.py         # Dataset loader with stratified split
│   │   │   ├── train.py           # Fine-tuning orchestrator
│   │   │   └── eval.py            # Model evaluation (planned)
│   │   └── tests/                 # Test data and validation
│   │       └── data/
│   │           ├── images/        # Test images
│   │           └── videos/        # Test videos
│   └── api/                       # REST API endpoints (planned)
├── main.py                        # Application entry point
├── pyproject.toml                 # UV package dependencies
└── README.md                      # This file
```

### Package Structure Notes

- All directories have `__init__.py` files for proper Python package imports
- Use **relative imports** throughout the codebase
- Run modules from the `src/` package root: `python -m model.finetune.train`
- The `finetune/` directory is the **canonical training pipeline** (no duplicate train folders)

## 🚀 Development Stages

### Stage 1: Foundation ✅ (Completed)

- [x] Project structure setup
- [x] VLM model integration (InternVL3-2B)
- [x] Multi-hardware support (CUDA/DirectML/CPU)
- [x] Basic video processing capabilities
- [x] Configuration management
- [x] Memory optimization and cleanup
- [x] UV package management setup

### Stage 2: Data Pipeline & Fine-tuning ✅ (Completed)

- [x] WLASL dataset integration
- [x] YouTube video downloader for dataset collection
- [x] Data filtering and preprocessing pipeline
- [x] Dataset generation for fine-tuning (320-word glossary)
- [x] Dataset loader with gloss-aware stratified train/test split
- [x] Training pipeline structure with SFT configuration
- [x] Package structure with proper `__init__.py` files and relative imports
- [x] **Custom video collation function with frame sampling**
- [x] **Label masking implementation (prompt tokens = -100, answer tokens = target)**
- [x] **PEFT/LoRA integration for efficient fine-tuning**
- [x] **Complete training loop with Hugging Face Trainer**
- [x] Hardware-specific optimizations (fp16 for CUDA, float32 for DirectML)

### Stage 3: API Development 📋 (Planned)

- [ ] REST API endpoints for image/video processing
- [ ] WebSocket integration for real-time streaming
- [ ] Request/response validation
- [ ] Error handling and logging
- [ ] Authentication and rate limiting

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
- **TRL (Transformer Reinforcement Learning)**: Fine-tuning with SFT
- **PEFT**: Parameter-Efficient Fine-Tuning

### Dataset & Processing

- **WLASL Dataset**: Word-Level American Sign Language dataset
- **Decord**: High-performance video frame extraction and sampling
- **Custom Data Pipeline**: Filtering and preprocessing for 320-word glossary
- **Stratified Splitting**: Gloss-aware train/test splits for balanced evaluation

### Hardware Acceleration

- **CUDA**: NVIDIA GPU acceleration
- **DirectML**: AMD GPU acceleration via PyTorch-DirectML
- **BitsAndBytes**: 4-bit quantization for memory efficiency
- **Accelerate**: Distributed training support

### Package Management & Environment

- **UV**: Ultra-fast Python package installer and resolver
- **YAML**: Configuration management
- **Python-dotenv**: Environment variable management

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- CUDA-compatible GPU (optional, for acceleration)
- AMD GPU with DirectML support (optional, for acceleration)
- UV package manager (recommended)

### Setup Instructions

1. **Install UV (if not already installed)**

   ```bash
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Alternative: via pip
   pip install uv
   ```

2. **Clone the repository**

   ```bash
   git clone https://github.com/Okeha/sign-language-detector-backend.git
   cd sign-language-detector-backend
   ```

3. **Set up Python environment and dependencies**

   ```bash
   # Create virtual environment and install dependencies
   uv sync

   # Activate the virtual environment
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

4. **Set up environment variables**

   ```bash
   # Create .env file
   echo "HF_TOKEN=your_hugging_face_token_here" > .env
   ```

5. **Test the base model**
   ```bash
   cd src/model
   python model_loader.py
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

### Dataset Setup and Fine-tuning Pipeline

#### Step 1: Download WLASL Dataset Videos

```bash
# Navigate to data engineering directory
cd src/model/finetune/data_engineering

# Download videos from WLASL dataset (requires WLASL JSON file)
python video_downloader.py
```

#### Step 2: Process and Filter Dataset

```bash
# Filter dataset by 320-word glossary and available videos
python filter_data.py
```

This will:

- Filter WLASL dataset to include only 320 core sign language words
- Check for downloaded videos in `raw_videos/` folder
- Generate training dataset in JSON format
- Save cleaned dataset to `datasets/wlasl_cleaned.json`

#### Step 3: Run Fine-tuning

```bash
# Navigate to src package root
cd C:\Users\okeha\Documents\projects\personal\signlanguagedetector\backend\src

# Run training as a module (proper way with relative imports)
python -m model.finetune.train
```

**Important**: Always run from the `src/` directory using the `-m` flag to ensure relative imports work correctly.

#### Dataset Structure

The processed dataset follows this format:

```json
{
  "prompt": "You are an expert sign-language recognition model. Identify the sign in the video and respond with exactly one word and nothing else.",
  "video_path": "raw_videos/12345.mp4",
  "gloss": "HELLO"
}
```

#### Supported Sign Language Words (320 Glossary)

The fine-tuning pipeline uses a carefully curated 320-word glossary covering:

- **Pronouns** (15 words): I, YOU, HE, SHE, THEY, etc.
- **Basic Verbs** (40 words): BE, HAVE, DO, GO, MAKE, etc.
- **Time Words** (25 words): NOW, TODAY, TOMORROW, etc.
- **People & Roles** (25 words): PERSON, FAMILY, TEACHER, etc.
- **Places** (25 words): HOME, SCHOOL, HOSPITAL, etc.
- **Objects** (30 words): BOOK, PHONE, FOOD, etc.
- **Feelings** (20 words): HAPPY, SAD, ANGRY, etc.
- **Descriptors** (35 words): BIG, SMALL, FAST, SLOW, etc.
- **Colors** (10 words): RED, BLUE, GREEN, etc.
- **Numbers** (20 words): ONE, TWO, THREE, etc.
- **Question Words** (10 words): WHO, WHAT, WHERE, etc.
- **Connectors** (10 words): AND, OR, BUT, etc.

#### Configuration

Modify `src/model/params/vlm.yml` to customize:

- Model selection
- Inference prompts
- Generation parameters

```yaml
model: OpenGVLab/InternVL3-2B-hf
prompt: You are a sign language interpreter. Given the video input, provide a concise and accurate text translation of the sign language being communicated.
```

#### Fine-tuning Configuration

The fine-tuning process uses PEFT with LoRA and custom data collation:

```python
# PEFT/LoRA Configuration
peft_config = LoraConfig(
    r=128,                          # LoRA rank
    lora_alpha=256,                 # LoRA scaling
    target_modules=[".*language_model.*attn.*"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./vlm_finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    fp16=(device_type == "cuda"),   # fp16 only for CUDA
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,     # Keep all dataset fields
    optim="adamw_torch",
)

# Custom data collator handles:
# - Video frame sampling (8 frames by default)
# - Processor formatting with <video> placeholder
# - Label masking (prompt=-100, answer=token_ids)
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

### Dataset Structure

```
src/model/finetune/data_engineering/
├── raw_videos/      # Downloaded WLASL videos (.mp4, .swf)
├── datasets/        # Processed datasets
│   └── wlasl_cleaned.json  # Filtered training dataset
└── raw_data/        # Original WLASL dataset files
```

### Running Tests

```bash
# Test model loading and inference
cd src
python -m model.model_loader

# Test data processing pipeline
cd model/finetune/data_engineering
python filter_data.py

# Test dataset loader with stratified split
cd ../../../src
python -c "from model.finetune.dataset import DatasetLoader; dl = DatasetLoader(); dl._train_test_split()"

# Test fine-tuning pipeline (once preprocessing is complete)
cd src
python -m model.finetune.train
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

#### Base Model

- **Model Loading**: ~15-30 seconds (depending on hardware)
- **Video Processing**: ~2-5 seconds per video (8 frames)
- **Memory Usage**: ~4-8GB VRAM (with 4-bit quantization)

#### Fine-tuning Process

- **Dataset Size**: 320 core sign language words from WLASL
- **Training Time**: Varies by dataset size and hardware
- **Memory Requirements**: ~8-16GB VRAM for training
- **Supported Batch Size**: 1-4 (depending on GPU memory)

### Optimization Features

- 4-bit quantization reduces memory usage by ~75%
- DirectML support for AMD GPUs
- Efficient memory cleanup and garbage collection
- Configurable generation parameters
- Gradient accumulation for effective larger batch sizes
- Parameter-efficient fine-tuning (PEFT) support

### Dataset Statistics

- **Total WLASL Vocabulary**: ~2,000+ words
- **Filtered Glossary**: 320 essential words
- **Video Sources**: YouTube via WLASL dataset
- **Processing Pipeline**: Automated filtering and dataset generation

## 🔮 Future Enhancements

### Short-term Goals

- Model evaluation and inference pipeline for trained checkpoints
- Hyperparameter optimization and experiment tracking
- REST API implementation with FastAPI
- WebSocket support for real-time processing
- Model serving and deployment infrastructure

### Long-term Vision

- Support for multiple sign languages (ASL, BSL, etc.)
- Advanced fine-tuning strategies (LoRA, QLoRA)
- Continuous learning from user feedback
- Mobile app integration
- Cloud deployment options
- Real-time video streaming optimization
- Custom model architectures for sign language

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
