"""
Dataset Quality Assessment Module

This module provides comprehensive dataset quality assessment functionality
including missing value analysis, duplicate detection, data imbalance calculation,
and overall quality scoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def assess_dataset_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive dataset quality assessment.
    
    Args:
        df: Input DataFrame to assess
        
    Returns:
        Dictionary containing comprehensive quality metrics and recommendations
    """
    try:
        quality_report = {
            'basic_metrics': calculate_basic_metrics(df),
            'missing_value_analysis': analyze_missing_values(df),
            'duplicate_analysis': analyze_duplicates(df),
            'data_imbalance_analysis': analyze_data_imbalance(df),
            'data_type_analysis': analyze_data_types(df),
            'outlier_analysis': analyze_outliers(df),
            'overall_quality_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall quality score
        quality_report['overall_quality_score'] = calculate_overall_quality_score(quality_report)
        
        # Generate recommendations
        quality_report['recommendations'] = generate_comprehensive_recommendations(quality_report)
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Failed to assess dataset quality: {str(e)}")
        return {
            'basic_metrics': {},
            'missing_value_analysis': {},
            'duplicate_analysis': {},
            'data_imbalance_analysis': {},
            'data_type_analysis': {},
            'outlier_analysis': {},
            'overall_quality_score': 0.0,
            'recommendations': ['Quality assessment failed'],
            'error': str(e)
        }


def calculate_basic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic dataset metrics."""
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_cells': len(df) * len(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'column_names': df.columns.tolist()
    }


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values in the dataset."""
    total_cells = len(df) * len(df.columns)
    missing_per_column = df.isnull().sum()
    total_missing = missing_per_column.sum()
    
    # Calculate missing value percentages per column
    missing_percentages = {}
    for column in df.columns:
        missing_count = missing_per_column[column]
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        missing_percentages[column] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2)
        }
    
    # Identify columns with high missing values
    high_missing_columns = [
        col for col, stats in missing_percentages.items() 
        if stats['percentage'] > 50
    ]
    
    moderate_missing_columns = [
        col for col, stats in missing_percentages.items() 
        if 10 < stats['percentage'] <= 50
    ]
    
    return {
        'total_missing_values': int(total_missing),
        'total_missing_percentage': round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0,
        'columns_with_missing': len([col for col in missing_per_column.index if missing_per_column[col] > 0]),
        'missing_per_column': missing_percentages,
        'high_missing_columns': high_missing_columns,
        'moderate_missing_columns': moderate_missing_columns,
        'complete_columns': [col for col, stats in missing_percentages.items() if stats['count'] == 0]
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze duplicate rows in the dataset."""
    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
    
    # Find duplicate rows with their indices
    duplicate_indices = df[duplicate_mask].index.tolist()
    
    # Analyze duplicates by keeping first occurrence
    duplicate_groups = []
    if duplicate_count > 0:
        # Group duplicates together
        seen_rows = set()
        for idx, row in df.iterrows():
            row_tuple = tuple(row.values)
            if row_tuple in seen_rows and idx in duplicate_indices:
                # This is a duplicate
                continue
            elif df.duplicated(subset=df.columns, keep=False).iloc[idx]:
                # This row has duplicates
                duplicate_group = df[df.duplicated(subset=df.columns, keep=False) & 
                                   (df == row).all(axis=1)].index.tolist()
                if len(duplicate_group) > 1:
                    duplicate_groups.append(duplicate_group)
                seen_rows.add(row_tuple)
    
    return {
        'duplicate_rows': int(duplicate_count),
        'duplicate_percentage': round(duplicate_percentage, 2),
        'duplicate_indices': duplicate_indices[:100],  # Limit to first 100 for performance
        'unique_rows': len(df) - duplicate_count,
        'duplicate_groups_count': len(duplicate_groups),
        'has_duplicates': duplicate_count > 0
    }


def analyze_data_imbalance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data imbalance in categorical columns."""
    categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    imbalance_analysis = {}
    overall_imbalance_scores = []
    
    for column in categorical_columns:
        value_counts = df[column].value_counts()
        
        if len(value_counts) > 1:
            # Calculate imbalance ratio (min/max)
            max_count = value_counts.max()
            min_count = value_counts.min()
            imbalance_ratio = min_count / max_count if max_count > 0 else 0
            
            # Calculate entropy (measure of distribution uniformity)
            probabilities = value_counts / value_counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(value_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            imbalance_analysis[column] = {
                'unique_values': len(value_counts),
                'imbalance_ratio': round(imbalance_ratio, 3),
                'entropy': round(entropy, 3),
                'normalized_entropy': round(normalized_entropy, 3),
                'most_frequent_value': str(value_counts.index[0]),
                'most_frequent_count': int(value_counts.iloc[0]),
                'least_frequent_value': str(value_counts.index[-1]),
                'least_frequent_count': int(value_counts.iloc[-1]),
                'value_distribution': {str(k): int(v) for k, v in value_counts.head(10).items()}
            }
            
            overall_imbalance_scores.append(imbalance_ratio)
    
    # Calculate overall imbalance score
    overall_imbalance = np.mean(overall_imbalance_scores) if overall_imbalance_scores else 1.0
    
    return {
        'categorical_columns_count': len(categorical_columns),
        'column_imbalance_analysis': imbalance_analysis,
        'overall_imbalance_ratio': round(overall_imbalance, 3),
        'severely_imbalanced_columns': [
            col for col, stats in imbalance_analysis.items() 
            if stats['imbalance_ratio'] < 0.1
        ],
        'moderately_imbalanced_columns': [
            col for col, stats in imbalance_analysis.items() 
            if 0.1 <= stats['imbalance_ratio'] < 0.3
        ]
    }


def analyze_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data types and their distribution."""
    type_analysis = {}
    type_counts = {}
    
    for column in df.columns:
        # Determine detailed data type
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column]):
                data_type = 'integer'
            else:
                data_type = 'float'
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            data_type = 'datetime'
        elif pd.api.types.is_bool_dtype(df[column]):
            data_type = 'boolean'
        else:
            # Check if it could be converted to numeric
            try:
                pd.to_numeric(df[column], errors='raise')
                data_type = 'numeric_string'
            except:
                # Check if it could be datetime
                try:
                    pd.to_datetime(df[column], errors='raise')
                    data_type = 'datetime_string'
                except:
                    data_type = 'categorical'
        
        type_analysis[column] = {
            'current_type': str(df[column].dtype),
            'inferred_type': data_type,
            'unique_values': int(df[column].nunique()),
            'sample_values': df[column].dropna().head(5).tolist()
        }
        
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    
    return {
        'column_type_analysis': type_analysis,
        'type_distribution': type_counts,
        'numeric_columns': [col for col, info in type_analysis.items() 
                           if info['inferred_type'] in ['integer', 'float']],
        'categorical_columns': [col for col, info in type_analysis.items() 
                               if info['inferred_type'] == 'categorical'],
        'potential_conversions': [
            col for col, info in type_analysis.items() 
            if info['inferred_type'] in ['numeric_string', 'datetime_string']
        ]
    }


def analyze_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze outliers in numerical columns."""
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    outlier_analysis = {}
    
    for column in numeric_columns:
        col_data = df[column].dropna()
        
        if len(col_data) > 0:
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
            # Z-score method (using 3 standard deviations)
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                z_scores = np.abs((col_data - mean_val) / std_val)
                zscore_outliers = (z_scores > 3).sum()
            else:
                zscore_outliers = 0
            
            outlier_analysis[column] = {
                'iqr_outliers': int(iqr_outliers),
                'iqr_outlier_percentage': round((iqr_outliers / len(col_data)) * 100, 2),
                'zscore_outliers': int(zscore_outliers),
                'zscore_outlier_percentage': round((zscore_outliers / len(col_data)) * 100, 2),
                'iqr_bounds': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound)
                },
                'statistics': {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
            }
    
    return {
        'numeric_columns_analyzed': len(numeric_columns),
        'column_outlier_analysis': outlier_analysis,
        'columns_with_many_outliers': [
            col for col, stats in outlier_analysis.items() 
            if stats['iqr_outlier_percentage'] > 10
        ]
    }


def calculate_overall_quality_score(quality_report: Dict[str, Any]) -> float:
    """
    Calculate overall dataset quality score (0-100).
    
    Scoring factors:
    - Missing values (25%): Lower missing values = higher score
    - Duplicates (20%): Fewer duplicates = higher score  
    - Data imbalance (20%): Better balance = higher score
    - Outliers (15%): Fewer outliers = higher score
    - Data type consistency (20%): Better type consistency = higher score
    """
    try:
        # Missing values score (0-100)
        missing_pct = quality_report['missing_value_analysis']['total_missing_percentage']
        missing_score = max(0, 100 - missing_pct * 2)  # Penalize missing values
        
        # Duplicates score (0-100)
        duplicate_pct = quality_report['duplicate_analysis']['duplicate_percentage']
        duplicate_score = max(0, 100 - duplicate_pct * 3)  # Penalize duplicates more heavily
        
        # Imbalance score (0-100)
        imbalance_ratio = quality_report['data_imbalance_analysis']['overall_imbalance_ratio']
        imbalance_score = imbalance_ratio * 100  # Higher ratio is better
        
        # Outliers score (0-100)
        outlier_analysis = quality_report['outlier_analysis']['column_outlier_analysis']
        if outlier_analysis:
            avg_outlier_pct = np.mean([
                stats['iqr_outlier_percentage'] for stats in outlier_analysis.values()
            ])
            outlier_score = max(0, 100 - avg_outlier_pct * 2)
        else:
            outlier_score = 100
        
        # Data type consistency score (0-100)
        type_analysis = quality_report['data_type_analysis']
        potential_conversions = len(type_analysis['potential_conversions'])
        total_columns = quality_report['basic_metrics']['total_columns']
        type_consistency_ratio = 1 - (potential_conversions / total_columns) if total_columns > 0 else 1
        type_score = type_consistency_ratio * 100
        
        # Weighted overall score
        overall_score = (
            missing_score * 0.25 +
            duplicate_score * 0.20 +
            imbalance_score * 0.20 +
            outlier_score * 0.15 +
            type_score * 0.20
        )
        
        return round(overall_score, 1)
        
    except Exception as e:
        logger.error(f"Failed to calculate overall quality score: {str(e)}")
        return 0.0


def generate_comprehensive_recommendations(quality_report: Dict[str, Any]) -> List[str]:
    """Generate comprehensive recommendations based on quality analysis."""
    recommendations = []
    
    try:
        # Missing values recommendations
        missing_analysis = quality_report['missing_value_analysis']
        if missing_analysis['total_missing_percentage'] > 20:
            recommendations.append("High percentage of missing values detected - consider data collection improvements")
        elif missing_analysis['total_missing_percentage'] > 10:
            recommendations.append("Moderate missing values - implement imputation strategies")
        
        if missing_analysis['high_missing_columns']:
            recommendations.append(f"Consider removing columns with >50% missing values: {', '.join(missing_analysis['high_missing_columns'][:3])}")
        
        # Duplicates recommendations
        duplicate_analysis = quality_report['duplicate_analysis']
        if duplicate_analysis['duplicate_percentage'] > 10:
            recommendations.append("High number of duplicate rows - remove duplicates to improve data quality")
        elif duplicate_analysis['duplicate_percentage'] > 1:
            recommendations.append("Some duplicate rows detected - consider deduplication")
        
        # Imbalance recommendations
        imbalance_analysis = quality_report['data_imbalance_analysis']
        if imbalance_analysis['severely_imbalanced_columns']:
            recommendations.append("Severe class imbalance detected - consider resampling techniques")
        elif imbalance_analysis['moderately_imbalanced_columns']:
            recommendations.append("Moderate class imbalance - monitor model performance carefully")
        
        # Outliers recommendations
        outlier_analysis = quality_report['outlier_analysis']
        if outlier_analysis['columns_with_many_outliers']:
            recommendations.append(f"Many outliers detected in: {', '.join(outlier_analysis['columns_with_many_outliers'][:3])}")
        
        # Data type recommendations
        type_analysis = quality_report['data_type_analysis']
        if type_analysis['potential_conversions']:
            recommendations.append("Some columns may benefit from type conversion for better analysis")
        
        # Overall quality recommendations
        overall_score = quality_report['overall_quality_score']
        if overall_score >= 80:
            recommendations.append("Dataset quality is excellent for analysis")
        elif overall_score >= 60:
            recommendations.append("Dataset quality is good with minor improvements needed")
        elif overall_score >= 40:
            recommendations.append("Dataset quality is moderate - several improvements recommended")
        else:
            recommendations.append("Dataset quality is poor - significant preprocessing required")
        
        return recommendations if recommendations else ["Dataset analysis completed"]
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")
        return ["Unable to generate quality recommendations"]