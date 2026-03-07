# Data Preprocessing Engine

## Overview

The data preprocessing engine provides comprehensive data transformation capabilities for the AI Data Analyst Platform. It handles various preprocessing operations including null value handling, outlier removal, scaling, encoding, and dataset quality assessment.

## Features Implemented

### Core Preprocessing Operations

1. **Null Value Handling**
   - Row removal: Remove rows containing null values
   - Column removal: Remove columns with high null percentages
   - Value filling: Fill nulls using mean, median, mode, or constant values

2. **Outlier Detection and Removal**
   - IQR method: Remove outliers beyond 1.5 * IQR from Q1/Q3
   - Z-score method: Remove outliers beyond specified standard deviations

3. **Data Scaling**
   - StandardScaler: Transform to mean=0, std=1
   - MinMaxScaler: Transform to range [0, 1]

4. **Categorical Encoding**
   - Label encoding: Convert categories to numerical labels
   - One-hot encoding: Create binary columns for categories

### Dataset Quality Assessment

Comprehensive quality analysis including:

- **Missing Value Analysis**: Percentage and distribution of missing values
- **Duplicate Detection**: Identification of duplicate rows
- **Data Imbalance Analysis**: Class distribution analysis for categorical variables
- **Data Type Analysis**: Type inference and conversion recommendations
- **Outlier Analysis**: Statistical outlier detection using IQR and Z-score methods
- **Overall Quality Score**: 0-100 score based on multiple quality factors

### Quality Scoring Algorithm

The overall quality score (0-100) is calculated using weighted factors:
- Missing values (25%): Lower missing values = higher score
- Duplicates (20%): Fewer duplicates = higher score  
- Data imbalance (20%): Better balance = higher score
- Outliers (15%): Fewer outliers = higher score
- Data type consistency (20%): Better type consistency = higher score

## API Endpoints

### POST /preprocess
Performs preprocessing operations on datasets.

**Request Body:**
```json
{
  "session_id": "uuid-string",
  "operations": [
    {
      "type": "null_removal",
      "parameters": {
        "method": "drop_rows",
        "columns": ["column1", "column2"]
      }
    },
    {
      "type": "scaling",
      "parameters": {
        "method": "standard",
        "columns": ["numeric_col1", "numeric_col2"]
      }
    }
  ]
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "operations_completed": 2,
  "operations_failed": 0,
  "processed_dataset": {
    "s3_key": "datasets/session-id/processed.csv",
    "shape": {
      "rows": 950,
      "columns": 12
    }
  },
  "quality_metrics": {
    "overall_quality_score": 87.5,
    "recommendations": ["Dataset quality is good for analysis"]
  },
  "operation_results": [...]
}
```

### GET /quality/{sessionId}
Retrieves comprehensive quality assessment for a dataset.

**Response:**
```json
{
  "session_id": "uuid-string",
  "dataset_name": "data.csv",
  "dataset_type": "processed",
  "quality_report": {
    "basic_metrics": {
      "total_rows": 1000,
      "total_columns": 10,
      "total_cells": 10000
    },
    "missing_value_analysis": {
      "total_missing_percentage": 5.2,
      "columns_with_missing": 3,
      "high_missing_columns": [],
      "moderate_missing_columns": ["age"]
    },
    "duplicate_analysis": {
      "duplicate_rows": 15,
      "duplicate_percentage": 1.5,
      "has_duplicates": true
    },
    "data_imbalance_analysis": {
      "overall_imbalance_ratio": 0.65,
      "severely_imbalanced_columns": [],
      "moderately_imbalanced_columns": ["category"]
    },
    "overall_quality_score": 82.3,
    "recommendations": [
      "Some missing values detected - consider imputation strategies",
      "Few duplicate rows detected - consider removal",
      "Dataset quality is good for analysis"
    ]
  }
}
```

## Operation Types

### null_removal
- `method`: "drop_rows" | "drop_columns"
- `columns`: Array of column names (optional)
- `threshold`: Threshold for column dropping (0-1)

### null_filling
- `strategy`: "mean" | "median" | "mode" | "constant"
- `columns`: Array of column names (optional)
- `fill_value`: Value for constant strategy

### outlier_removal
- `method`: "iqr" | "zscore"
- `columns`: Array of column names (optional)
- `threshold`: IQR multiplier or Z-score threshold

### scaling
- `method`: "standard" | "minmax"
- `columns`: Array of column names (optional)

### label_encoding
- `columns`: Array of column names (optional)

### one_hot_encoding
- `columns`: Array of column names (optional)
- `drop_first`: Boolean to avoid multicollinearity

## Property-Based Testing

The preprocessing engine includes comprehensive property-based tests validating:

- **Property 10**: Null value preprocessing preserves data integrity
- **Property 11**: Outlier removal accuracy using IQR and Z-score methods
- **Property 12**: Data scaling transformations maintain mathematical properties
- **Property 13**: Categorical encoding consistency and correctness
- **Property 14**: Operation persistence and logging completeness

## Dependencies

- pandas==2.1.4
- numpy==1.24.3
- scikit-learn==1.3.2
- boto3==1.34.0

## Files

- `lambda_function.py`: Main Lambda handler
- `quality_assessment.py`: Quality assessment module
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Integration

The preprocessing engine integrates with:
- **S3**: Dataset storage and retrieval
- **DynamoDB**: Operation logging and session management
- **API Gateway**: REST API endpoints
- **CloudWatch**: Monitoring and logging

## Error Handling

Comprehensive error handling includes:
- Input validation
- Operation-specific error recovery
- Detailed error logging
- Graceful degradation for partial failures