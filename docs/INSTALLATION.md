# CTOD Installation and Deployment Guide

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.7 or higher (3.8+ recommended)
- **RAM**: 8GB minimum, 16GB+ recommended for full dataset
- **Storage**: 50GB+ free space for complete dataset and outputs
- **Internet**: Stable connection for data downloads and API calls

### Recommended Requirements
- **RAM**: 32GB+ for full-scale processing
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for BERT models)
- **CPU**: Multi-core processor (8+ cores recommended)
- **Storage**: SSD with 100GB+ free space

## 📦 Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/chufangao/CTOD.git
cd CTOD

# 2. Create virtual environment
python -m venv ctod_env
source ctod_env/bin/activate  # On Windows: ctod_env\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch, transformers, snorkel; print('✓ Installation successful')"
```

### Method 2: Conda Installation

```bash
# 1. Create conda environment
conda create -n ctod python=3.8
conda activate ctod

# 2. Clone repository
git clone https://github.com/chufangao/CTOD.git
cd CTOD

# 3. Install core dependencies via conda
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Method 3: Docker Installation

```bash
# 1. Create Dockerfile (if not provided)
cat > Dockerfile << EOF
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["bash"]
EOF

# 2. Build and run container
docker build -t ctod .
docker run -it -v $(pwd):/app ctod
```

## 🔑 API Keys Setup

### Required API Keys

1. **NCBI API Key** (for PubMed access)
   - Create account at [NCBI](https://www.ncbi.nlm.nih.gov/account/)
   - Generate API key in account settings
   - Rate limit: 10 requests/second with key

2. **OpenAI API Key** (for LLM predictions)
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Generate API key in dashboard
   - Ensure sufficient credits (~$50-100 for full dataset)

3. **SerpAPI Key** (optional, for enhanced news scraping)
   - Register at [SerpAPI](https://serpapi.com/)
   - Get API key from dashboard
   - Free tier available with limited searches

### Environment Configuration

#### Option 1: Environment Variables
```bash
export NCBI_API_KEY="your_ncbi_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
export SERPAPI_KEY="your_serpapi_key_here"
```

#### Option 2: .env File
```bash
# Create .env file in project root
cat > .env << EOF
NCBI_API_KEY=your_ncbi_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_KEY=your_serpapi_key_here
EOF

# Load in Python scripts
from dotenv import load_dotenv
load_dotenv()
```

## 💾 Data Setup

### Required Datasets

#### 1. CTTI Clinical Trial Data

**Automated Download:**
```bash
python download_ctti.py
# Downloads to ./downloads/CTTI_new.zip
```

**Manual Download:**
1. Visit [CTTI Download Page](https://aact.ctti-clinicaltrials.org/download)
2. Download "Pipe-delimited files" (latest version)
3. Extract to desired directory
4. Note the path for configuration

#### 2. TOP Dataset (for evaluation)

```bash
git clone https://github.com/futianfan/clinical-trial-outcome-prediction
# Reference ground truth labels for comparison
```

#### 3. FDA Orange Book (for trial linkage)

**Automated Download:**
```bash
cd clinical_trial_linkage
python download_data.py --save_path /path/to/save
```

**Manual Download:**
1. Download from [FDA Orange Book](https://www.fda.gov/media/76860/download?attachment)
2. Save to `clinical_trial_linkage/FDA_approvals/`

#### 4. Pre-computed Results (Optional)

Download from [Zenodo](https://doi.org/10.57967/hf/4597) for faster setup:
- Pre-scraped news headlines
- Pre-computed embeddings
- Sample processed data

## 🏃‍♂️ Quick Deployment

### Development Setup (5 minutes)

```bash
# Minimal setup for exploration
git clone https://github.com/chufangao/CTOD.git
cd CTOD
pip install -r requirements.txt

# Test with tutorials
cd tutorials
jupyter notebook getting_started_cto_vs_top.ipynb
```

### Production Setup (Full Pipeline)

```bash
# 1. Complete installation
git clone https://github.com/chufangao/CTOD.git
cd CTOD
python -m venv ctod_env
source ctod_env/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env  # Edit with your API keys

# 3. Download data
python download_ctti.py

# 4. Run complete pipeline
export DATA_PATH="/path/to/ctti/data"
export SAVE_PATH="/path/to/results"
bash pipeline.sh
```

## 🔧 Configuration Management

### Directory Structure Setup

```bash
# Recommended directory structure
mkdir -p ctod_workspace/{data,results,logs}
cd ctod_workspace

# Data directories
mkdir -p data/{ctti,top,fda,external}

# Results directories  
mkdir -p results/{embeddings,labels,models,evaluation}

# Clone CTOD
git clone https://github.com/chufangao/CTOD.git
```

### Configuration Files

#### `config.yaml` (Create this for centralized config)

```yaml
# CTOD Configuration File
data_paths:
  ctti: "/path/to/ctti/data"
  top: "/path/to/top/dataset"
  fda: "/path/to/fda/data"
  
save_paths:
  embeddings: "/path/to/save/embeddings"
  labels: "/path/to/save/labels"
  models: "/path/to/save/models"

api_keys:
  ncbi: "${NCBI_API_KEY}"
  openai: "${OPENAI_API_KEY}"
  serpapi: "${SERPAPI_KEY}"

processing:
  num_workers: 4
  gpu_ids: [0, 1]
  batch_size: 16
  max_trials: null  # null for all trials, number for testing

labeling:
  quantile_thresholds:
    phase1: 0.6
    phase2: 0.7
    phase3: 0.8
  aggregation_method: "random_forest"  # or "snorkel"
```

## 🚀 Deployment Options

### Option 1: Local Development

```bash
# Activate environment
source ctod_env/bin/activate

# Run specific modules
cd llm_prediction_on_pubmed
python extract_pubmed_abstracts.py --config ../config.yaml
```

### Option 2: Server Deployment

```bash
# Setup on remote server
ssh user@server
git clone https://github.com/chufangao/CTOD.git
cd CTOD

# Use screen/tmux for long-running processes
screen -S ctod_pipeline
bash pipeline.sh
# Ctrl+A, D to detach
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

```bash
# Example for AWS EC2
# 1. Launch EC2 instance (g4dn.xlarge or larger for GPU)
# 2. Install dependencies
sudo apt update
sudo apt install python3-pip git
pip3 install -r requirements.txt

# 3. Configure AWS credentials for data storage
aws configure
aws s3 sync s3://your-data-bucket ./data/

# 4. Run pipeline with cloud storage
python your_script.py --save_path s3://your-results-bucket/
```

## 📊 Monitoring and Logging

### Progress Tracking

```python
# Add progress tracking to long-running processes
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_with_monitoring(trials):
    """Process trials with progress monitoring."""
    
    results = []
    for trial in tqdm(trials, desc="Processing trials"):
        try:
            result = process_single_trial(trial)
            results.append(result)
            
            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)} trials successfully")
                
        except Exception as e:
            logger.error(f"Failed to process {trial['nct_id']}: {str(e)}")
            continue
    
    return results
```

### Resource Monitoring

```bash
# Monitor system resources during processing
htop          # CPU and memory usage
nvidia-smi    # GPU usage (if applicable)
df -h         # Disk space usage
```

## 🐛 Troubleshooting Installation

### Common Issues and Solutions

#### Python Version Conflicts
```bash
# Check Python version
python --version

# Use specific Python version
python3.8 -m venv ctod_env
```

#### Package Installation Failures
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install with verbose output for debugging
pip install -v transformers

# Use conda for problematic packages
conda install pytorch torchvision torchaudio
pip install transformers
```

#### CUDA/GPU Setup Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
```bash
# Monitor memory usage
free -h

# Increase swap space (Linux)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Network/API Issues
```bash
# Test API connectivity
curl -X GET "https://api.openai.com/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test NCBI API
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&api_key=$NCBI_API_KEY"
```

### Performance Optimization

#### For Limited Resources
```bash
# Run with development flag (smaller dataset)
python script_name.py --dev

# Reduce batch sizes
python script_name.py --batch_size 8

# Use CPU instead of GPU if memory limited
python script_name.py --device cpu
```

#### For High-Performance Setup
```bash
# Use multiple GPUs
python script_name.py --gpu_ids 0,1,2,3

# Increase workers
python script_name.py --num_workers 8

# Enable mixed precision training
python script_name.py --fp16
```

## 📋 Verification Checklist

After installation, verify your setup:

- [ ] **Dependencies**: All packages install without errors
- [ ] **API Keys**: All required API keys are configured and working
- [ ] **Data Access**: Can download and access CTTI data
- [ ] **GPU Support**: CUDA working (if using GPU)
- [ ] **Tutorials**: Can run getting_started notebook successfully
- [ ] **Modules**: Can import core CTOD modules without errors

### Quick Verification Script

```python
# Run this script to verify installation
def verify_ctod_installation():
    """Verify CTOD installation and dependencies."""
    
    checks = {}
    
    # Check imports
    try:
        import torch, transformers, snorkel, pandas, sklearn
        checks['dependencies'] = True
    except ImportError as e:
        checks['dependencies'] = f"Missing: {e.name}"
    
    # Check GPU
    try:
        import torch
        checks['gpu'] = torch.cuda.is_available()
    except:
        checks['gpu'] = False
    
    # Check API keys (if set)
    import os
    checks['apis'] = {
        'ncbi': bool(os.getenv('NCBI_API_KEY')),
        'openai': bool(os.getenv('OPENAI_API_KEY'))
    }
    
    return checks

if __name__ == "__main__":
    results = verify_ctod_installation()
    print("CTOD Installation Verification:")
    for check, result in results.items():
        print(f"  {check}: {result}")
```

## 🆘 Getting Help

### Before Asking for Help

1. **Check this documentation** for solutions
2. **Search existing issues** on GitHub
3. **Try with `--dev` flag** for testing with smaller datasets
4. **Check logs** for specific error messages

### Where to Get Help

- **GitHub Issues**: [Open an issue](https://github.com/chufangao/CTOD/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chufangao/CTOD/discussions)
- **Email**: Contact maintainers for urgent issues

### Information to Include in Help Requests

```bash
# System information
python --version
pip freeze > installed_packages.txt

# Error logs
python your_failing_script.py 2>&1 | tee error_log.txt

# System specs
free -h  # Memory
df -h    # Disk space
nvidia-smi  # GPU info (if applicable)
```