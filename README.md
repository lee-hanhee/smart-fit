# Smart-Fit: Intelligent Exercise Tracking System

Smart-Fit is an advanced exercise tracking and analysis system that uses machine learning to monitor and analyze barbell exercises. The system processes motion data to track repetitions, detect exercise patterns, and provide insights into workout performance.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Exercise Repetition Counting**: Automatically counts repetitions during barbell exercises
- **Motion Data Analysis**: Processes and analyzes motion data to identify exercise patterns
- **Outlier Detection**: Identifies and removes anomalous data points for accurate analysis
- **Temporal and Frequency Analysis**: Provides both time-domain and frequency-domain analysis of exercise movements
- **Machine Learning Integration**: Uses advanced algorithms to classify and predict exercise patterns
- **Data Visualization**: Includes tools for visualizing exercise data and analysis results

## Installation

### Prerequisites

- Python 3.8.15
- Conda package manager

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/smart-fit.git
cd smart-fit
```

2. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate tracking-barbell-exercises
```

3. Verify the installation:

```bash
python -c "import numpy; import pandas; import matplotlib"
```

## Usage

The project is organized into several key components:

1. **Data Processing**:

   - Use `src/features/remove_outliers.py` to clean motion data
   - Apply `src/features/count_repetitions.py` to track exercise repetitions
   - Utilize `src/features/build_features.py` for feature engineering

2. **Model Training**:

   - Run `src/models/train_model.py` to train the exercise classification model
   - Use `src/models/LearningAlgorithms.py` for custom learning algorithms

3. **Visualization**:
   - Access visualization tools in the `src/visualization` directory

## Project Structure

```
smart-fit/
├── data/                  # Raw and processed data
├── report/               # Analysis reports and visualizations
├── src/                  # Source code
│   ├── data/            # Data processing utilities
│   ├── features/        # Feature engineering modules
│   ├── models/          # Machine learning models
│   └── visualization/   # Data visualization tools
├── environment.yml      # Conda environment configuration
├── LICENSE              # Project license
└── README.md           # Project documentation
```

## Technologies Used

- **Python 3.8.15**: Core programming language
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Jupyter**: Interactive development environment
- **Machine Learning**: Custom learning algorithms for exercise classification

## Contributing

We welcome contributions to Smart-Fit! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Note

This project is currently under active development. Some features may be incomplete or subject to change. We welcome feedback and contributions from the community.
