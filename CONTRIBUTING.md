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

## 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **System information** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Sample data** if applicable (anonymized)

## 💡 Feature Requests

For new features, please:

- **Check existing issues** to avoid duplicates
- **Describe the problem** your feature would solve
- **Propose a solution** with implementation details
- **Consider backward compatibility**
- **Provide use cases** and examples

## 📋 Types of Contributions

We welcome various types of contributions:

### 🔧 Code Contributions
- Bug fixes
- New labeling functions
- Performance improvements
- New baseline models
- Code refactoring

### 📚 Documentation
- README improvements
- Code documentation (docstrings)
- Tutorial notebooks
- API documentation
- Usage examples

### 🧪 Testing
- Unit tests for existing functions
- Integration tests for pipelines
- Performance benchmarks
- Data validation tests

### 📊 Data & Models
- New datasets integration
- Model performance improvements
- Evaluation metrics
- Visualization tools

## 🎯 Project Priorities

Current high-priority areas for contribution:

1. **Testing Infrastructure**: Add comprehensive unit tests
2. **Performance Optimization**: Improve processing speed for large datasets
3. **Documentation**: Expand API documentation and examples
4. **Model Evaluation**: Add more baseline models and evaluation metrics
5. **Data Pipeline**: Improve error handling and data validation

## 📝 Pull Request Process

1. **Ensure your PR:**
   - Has a clear title and description
   - Includes tests for new functionality
   - Updates relevant documentation
   - Passes all existing tests
   - Follows code style guidelines

2. **PR Review Process:**
   - Maintainers will review your PR
   - Address feedback and make requested changes
   - Keep your branch up to date with main
   - Be patient during the review process

3. **After Approval:**
   - Your PR will be merged by maintainers
   - Delete your feature branch
   - Update your local repository

## 🤝 Community Guidelines

- **Be respectful** and inclusive in all interactions
- **Help others** learn and contribute
- **Follow the [Code of Conduct](./CODE_OF_CONDUCT.md)**
- **Ask questions** if you're unsure about anything

## 📞 Getting Help

- **Issues**: Open an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact maintainers for sensitive issues

## 🏷️ Licensing

By contributing to CTOD, you agree that your contributions will be licensed under the same [MIT License](./LICENSE) that covers the project.

---

Thank you for contributing to CTOD! Your efforts help advance clinical trial outcome prediction research. 🙏