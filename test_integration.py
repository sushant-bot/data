#!/usr/bin/env python3
"""
Integration test for the AI Data Analyst Platform data processing pipeline.
Tests the complete workflow from upload to preprocessing.
"""

import pandas as pd
import json
import base64
import sys
import os

# Add lambda directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lambda', 'upload'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lambda', 'preview'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lambda', 'processing'))

from unittest.mock import Mock, patch

def test_integration_workflow():
    """Test the complete data processing workflow."""
    print("Testing AI Data Analyst Platform Integration...")
    
    # Create test dataset
    test_data = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
        'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
        'mixed_col': [1.1, 2.2, None, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
    })
    
    csv_string = test_data.to_csv(index=False)
    encoded_content = base64.b64encode(csv_string.encode()).decode()
    
    # Test 1: Upload Lambda
    print("1. Testing Upload Lambda...")
    try:
        from lambda_function import lambda_handler as upload_handler, calculate_dataset_statistics, detect_pii_with_guardrails
        
        # Test dataset statistics calculation
        stats = calculate_dataset_statistics(test_data)
        assert stats['row_count'] == 10
        assert stats['column_count'] == 3
        assert 'numeric_col' in stats['data_types']
        assert stats['missing_values']['numeric_col'] == 1
        print("   ✓ Dataset statistics calculation works")
        
        # Test PII detection
        pii_results = detect_pii_with_guardrails(test_data)
        assert 'pii_detected' in pii_results
        print("   ✓ PII detection works")
        
    except Exception as e:
        print(f"   ✗ Upload Lambda test failed: {e}")
        return False
    
    # Test 2: Preview Lambda
    print("2. Testing Preview Lambda...")
    try:
        # Clear the module cache and import preview lambda
        if 'lambda_function' in sys.modules:
            del sys.modules['lambda_function']
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda', 'preview'))
        import lambda_function as preview_lambda
        
        # Test preview generation
        preview = preview_lambda.generate_dataset_preview(test_data)
        assert preview['total_rows_available'] == 10
        assert len(preview['columns']) == 3
        assert len(preview['rows']) <= 10
        print("   ✓ Dataset preview generation works")
        
        # Test detailed statistics
        detailed_stats = preview_lambda.calculate_detailed_statistics(test_data)
        assert 'column_statistics' in detailed_stats
        assert 'missing_value_summary' in detailed_stats
        print("   ✓ Detailed statistics calculation works")
        
    except Exception as e:
        print(f"   ✗ Preview Lambda test failed: {e}")
        return False
    
    # Test 3: Processing Lambda
    print("3. Testing Processing Lambda...")
    try:
        # Clear the module cache and import processing lambda
        if 'lambda_function' in sys.modules:
            del sys.modules['lambda_function']
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda', 'processing'))
        import lambda_function as processing_lambda
        from quality_assessment import assess_dataset_quality
        
        # Test null filling
        operation = {
            'type': 'null_filling',
            'parameters': {
                'strategy': 'mean',
                'columns': ['numeric_col', 'mixed_col']
            }
        }
        
        processed_df, result = processing_lambda.execute_preprocessing_operation(test_data, operation)
        assert processed_df['numeric_col'].isnull().sum() == 0
        assert result['strategy_used'] == 'mean'
        print("   ✓ Null filling operation works")
        
        # Test scaling
        scaling_operation = {
            'type': 'scaling',
            'parameters': {
                'method': 'standard',
                'columns': ['numeric_col', 'mixed_col']
            }
        }
        
        scaled_df, scaling_result = processing_lambda.execute_preprocessing_operation(processed_df, scaling_operation)
        assert scaling_result['method_used'] == 'standard'
        print("   ✓ Scaling operation works")
        
        # Test quality assessment
        quality_report = assess_dataset_quality(test_data)
        assert 'overall_quality_score' in quality_report
        assert 'recommendations' in quality_report
        assert isinstance(quality_report['overall_quality_score'], (int, float))
        print("   ✓ Quality assessment works")
        
    except Exception as e:
        print(f"   ✗ Processing Lambda test failed: {e}")
        return False
    
    print("\n✅ All integration tests passed!")
    print("Data processing pipeline is working correctly.")
    return True

if __name__ == "__main__":
    success = test_integration_workflow()
    sys.exit(0 if success else 1)