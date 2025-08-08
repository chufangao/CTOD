# Contributing to CTOD

We welcome contributions to the Clinical Trial Outcome Detection (CTOD) project! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Familiarity with clinical trial data and machine learning concepts

### Setting Up Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/CTOD.git
   cd CTOD
   ```

3. **Set up development environment:**
   ```bash
   # Create virtual environment
   python -m venv ctod_env
   source ctod_env/bin/activate  # On Windows: ctod_env\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install jupyter matplotlib seaborn  # For development
   ```

4. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/chufangao/CTOD.git
   ```

## 🔄 Development Workflow

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, documented code
   - Add docstrings to new functions
   - Follow existing code style
   - Test your changes thoroughly

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add meaningful commit message describing your changes"
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

## 📚 Documentation

### Code Style Guidelines

- **Python Code:**
  - Follow PEP 8 style guidelines
  - Use meaningful variable and function names
  - Add docstrings for all public functions and classes
  - Include type hints where appropriate

- **Documentation:**
  - Use clear, concise language
  - Include code examples where helpful
  - Update README files when adding new features
  - Follow markdown best practices

### Example Function Documentation

```python
def extract_trial_outcomes(data_path: str, trial_ids: list) -> pd.DataFrame:
    """
    Extract clinical trial outcome labels from CTTI data.
    
    Args:
        data_path (str): Path to CTTI dataset directory
        trial_ids (list): List of NCT IDs to process
        
    Returns:
        pd.DataFrame: DataFrame with trial IDs and outcome labels
        
    Raises:
        FileNotFoundError: If CTTI data files are not found
        ValueError: If invalid trial IDs are provided
        
    Example:
        >>> outcomes = extract_trial_outcomes('/data/ctti', ['NCT12345'])
        >>> print(outcomes.head())
    """
    # Implementation here
    pass
```

## 🔗 Integration

For detailed technical information, see our [docs directory](./docs/):
- [API Documentation](./docs/API.md)
- [Architecture Overview](./docs/ARCHITECTURE.md)  
- [Installation Guide](./docs/INSTALLATION.md)

Thank you for contributing to CTOD! 🙏