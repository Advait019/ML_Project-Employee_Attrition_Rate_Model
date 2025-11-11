# Employee Attrition Prediction ML Project

## Overview

This repository contains a comprehensive machine learning analysis for predicting employee attrition. The project combines supervised classification techniques with business-focused policy simulation and cost-benefit analysis. It addresses class imbalance through SMOTE and threshold optimization, provides rigorous validation with stratified splits and fixed random seeds, and includes bootstrap-based confidence intervals for statistical robustness. The project is designed for academic evaluation and emphasizes reproducibility, technical rigor, and practical business impact.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Project Structure
- **Single Notebook Architecture**: The core analysis resides in `churned_improved.ipynb`, a Jupyter notebook that contains all data processing, model training, validation, and analysis steps
- **Documentation-First Approach**: The project emphasizes comprehensive documentation with `JUSTIFICATION.md` explaining methodology and `README.md` providing setup instructions

### Machine Learning Pipeline
1. **Data Validation**: Fixed random seed (RANDOM_STATE = 42) for reproducibility
2. **Train/Test Split**: Stratified 80/20 split to preserve class distribution
3. **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) for balancing training data
4. **Model Training**: Multiple classifiers including XGBoost with hyperparameter tuning
5. **Evaluation**: Bootstrap-based confidence intervals, threshold optimization, and comprehensive metrics
6. **Business Analysis**: Cost-benefit calculations and policy simulation

### Key Design Decisions

**Problem Addressed**: Employee attrition prediction is inherently imbalanced (fewer employees leave than stay), requiring specialized handling to avoid biased models.

**Solution**: Multi-pronged approach combining SMOTE resampling, class weight adjustments, and threshold optimization to balance precision/recall for business decisions.

**Rationale**: Standard ML approaches fail on imbalanced datasets. This architecture ensures both statistical validity (through stratification and CIs) and business value (through cost analysis).

**Pros**: Rigorous, reproducible, and business-relevant. Addresses academic evaluation criteria comprehensively.

**Cons**: Single-notebook architecture may become unwieldy for larger datasets or production deployment. No modular code structure for reusability.

### Reproducibility Framework
- **Fixed Seeds**: All random operations use RANDOM_STATE = 42 for deterministic results
- **Environment Specification**: Complete `requirements.txt` with pinned versions
- **Validation Protocol**: Documented train/test splits with explicit ratios
- **Statistical Rigor**: Bootstrap confidence intervals for uncertainty quantification

## External Dependencies

### Python Libraries (ML & Data Science)
- **pandas (1.5.3)**: Data manipulation and analysis
- **numpy (1.24.3)**: Numerical computing and array operations
- **scikit-learn (1.2.2)**: Core ML algorithms, preprocessing, and metrics
- **xgboost (1.7.5)**: Gradient boosting classifier for high-performance modeling
- **imbalanced-learn (0.10.1)**: SMOTE and class imbalance handling techniques
- **scipy (1.10.1)**: Statistical computations and bootstrap methods

### Visualization
- **matplotlib (3.7.1)**: Base plotting library for figures
- **seaborn (0.12.2)**: Statistical visualizations and enhanced aesthetics

### Development Environment
- **jupyter (1.0.0)**: Notebook interface for interactive analysis
- **ipykernel (6.22.0)**: Jupyter kernel for Python execution
- **tqdm (4.65.0)**: Progress bars for long-running operations

### Data Sources
- Employee attrition dataset (source not specified in repository, likely loaded within notebook)
- No external APIs or databases; analysis is self-contained within the notebook

### Academic Context
- Designed for Big Data Analytics course evaluation at Goa Institute of Management
- Targets academic scoring criteria including reproducibility (1.0/4 originally), technical quality (2.8/4), and overall rigor
- Expected improvement to 26-28/30 score after implementing documented fixes