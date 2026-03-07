"""
Property-based tests for AI Assistant Lambda function.

These tests validate the correctness properties defined in the design document
for the AI Data Analyst Platform AI recommendation functionality.

Properties tested:
- Property 21: AI Recommendation Generation
- Property 22: AI Decision Persistence
- Property 26: AI Response Caching
- Property 29: Quality-Based AI Recommendations
"""

import pytest
import json
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal

# Set AWS region before importing lambda (boto3 clients initialize at module level)
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'testing')

# Load ai_assistant lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ai_assistant_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'ai_assistant', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['ai_assistant_lambda'] = _module

lambda_handler = _module.lambda_handler
analyze_dataset_characteristics = _module.analyze_dataset_characteristics
build_recommendation_prompt = _module.build_recommendation_prompt
generate_prompt_hash = _module.generate_prompt_hash
check_cache = _module.check_cache
store_cache = _module.store_cache
invoke_bedrock_with_retry = _module.invoke_bedrock_with_retry
parse_ai_response = _module.parse_ai_response
generate_rule_based_recommendations = _module.generate_rule_based_recommendations
generate_quality_based_recommendations = _module.generate_quality_based_recommendations
store_ai_decision = _module.store_ai_decision
create_response = _module.create_response


# Test data generators
@st.composite
def dataset_with_target(draw):
    """Generate a DataFrame with a clear classification target column."""
    num_rows = draw(st.integers(min_value=20, max_value=100))
    num_numeric = draw(st.integers(min_value=2, max_value=5))

    data = {}
    for i in range(num_numeric):
        data[f"feature_{i}"] = draw(st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        ))

    # Add a classification target
    num_classes = draw(st.integers(min_value=2, max_value=5))
    data['target'] = draw(st.lists(
        st.sampled_from([f"class_{i}" for i in range(num_classes)]),
        min_size=num_rows, max_size=num_rows
    ))

    return pd.DataFrame(data)


@st.composite
def dataset_without_target(draw):
    """Generate a DataFrame with only numeric columns with high cardinality (no clear target)."""
    num_rows = draw(st.integers(min_value=50, max_value=100))
    num_cols = draw(st.integers(min_value=2, max_value=6))

    data = {}
    for i in range(num_cols):
        # Use floats to ensure high unique count (won't be detected as classification target)
        data[f"metric_{i}"] = draw(st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        ))

    return pd.DataFrame(data)


@st.composite
def dataset_with_missing(draw):
    """Generate a DataFrame with missing values."""
    num_rows = draw(st.integers(min_value=20, max_value=80))
    data = {}
    for i in range(3):
        vals = draw(st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=num_rows, max_size=num_rows
        ))
        # Introduce missing values
        missing_count = draw(st.integers(min_value=1, max_value=max(1, num_rows // 5)))
        indices = draw(st.lists(
            st.integers(min_value=0, max_value=num_rows - 1),
            min_size=missing_count, max_size=missing_count, unique=True
        ))
        for idx in indices:
            vals[idx] = np.nan
        data[f"col_{i}"] = vals

    data['target'] = draw(st.lists(
        st.sampled_from(['A', 'B']),
        min_size=num_rows, max_size=num_rows
    ))

    return pd.DataFrame(data)


@st.composite
def quality_report_data(draw):
    """Generate a quality report dictionary for testing."""
    overall_score = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    overall_missing_pct = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    dup_count = draw(st.integers(min_value=0, max_value=50))
    imbalance_ratio = draw(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False))

    return {
        'overall_quality_score': overall_score,
        'missing_value_analysis': {
            'overall_missing_percentage': overall_missing_pct
        },
        'duplicate_analysis': {
            'duplicate_count': dup_count
        },
        'data_imbalance_analysis': {
            'max_imbalance_ratio': imbalance_ratio
        }
    }


class TestAIRecommendationGeneration:
    """
    Property 21: AI Recommendation Generation
    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
    """

    @given(df=dataset_with_target())
    @settings(max_examples=20)
    def test_property_21_classification_recommendations(self, df):
        """
        Feature: ai-data-analyst-platform, Property 21: AI Recommendation Generation

        For any uploaded dataset with a classification target, the AI Assistant should
        generate appropriate model recommendations with confidence scores and reasoning.
        """
        characteristics = analyze_dataset_characteristics(df)
        recommendations = generate_rule_based_recommendations(characteristics, None)

        # Must have recommended models
        assert len(recommendations['recommended_models']) > 0

        # Each model must have required fields
        for model in recommendations['recommended_models']:
            assert 'model' in model
            assert 'confidence' in model
            assert 'reasoning' in model
            assert 0.0 <= model['confidence'] <= 1.0
            assert len(model['reasoning']) > 0

        # Must have analysis type
        assert recommendations['analysis_type'] in [
            'supervised_classification', 'supervised_regression',
            'unsupervised_clustering', 'exploratory'
        ]

        # Must have reasoning
        assert 'reasoning' in recommendations
        assert len(recommendations['reasoning']) > 0

        # Source must be identified
        assert recommendations['source'] == 'rule_based_fallback'

    @given(df=dataset_without_target())
    @settings(max_examples=15)
    def test_property_21_unsupervised_recommendations(self, df):
        """
        For datasets without a clear target, AI should recommend unsupervised methods.
        """
        characteristics = analyze_dataset_characteristics(df)
        # Only test when no potential targets are detected
        assume(len(characteristics['potential_targets']) == 0)

        recommendations = generate_rule_based_recommendations(characteristics, None)

        assert len(recommendations['recommended_models']) > 0
        assert recommendations['analysis_type'] in ['unsupervised_clustering', 'exploratory']

        # Should recommend clustering algorithms
        model_names = [m['model'] for m in recommendations['recommended_models']]
        assert any(m in model_names for m in ['kmeans', 'dbscan'])

    @given(df=dataset_with_missing())
    @settings(max_examples=15)
    def test_property_21_preprocessing_recommendations(self, df):
        """
        For datasets with missing values, preprocessing recommendations should be generated.
        """
        characteristics = analyze_dataset_characteristics(df)
        assert characteristics['has_missing_values'] is True

        recommendations = generate_rule_based_recommendations(characteristics, None)
        assert len(recommendations['recommended_preprocessing']) > 0

        # Should recommend null handling
        steps = [p['step'] for p in recommendations['recommended_preprocessing']]
        assert any('null' in s or 'fill' in s for s in steps)

    def test_dataset_characteristics_analysis(self):
        """Dataset characteristics should accurately reflect the data."""
        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [5.0, 4.0, 3.0, 2.0, 1.0],
            'cat1': ['a', 'b', 'a', 'b', 'a'],
            'target': [0, 1, 0, 1, 0]
        })

        chars = analyze_dataset_characteristics(df)

        assert chars['num_rows'] == 5
        assert chars['num_columns'] == 4
        assert chars['numeric_columns'] == 3  # num1, num2, target
        assert chars['categorical_columns'] == 1  # cat1
        assert chars['has_missing_values'] is False
        assert chars['has_categorical'] is True
        assert chars['size_class'] == 'small'


class TestAIDecisionPersistence:
    """
    Property 22: AI Decision Persistence
    Validates: Requirements 8.7, 14.1, 14.2, 14.3, 14.4
    """

    def setup_method(self):
        self.dynamodb_mock = Mock()
        self.table_mock = Mock()
        self.dynamodb_mock.Table.return_value = self.table_mock
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.dynamodb_patcher.stop()

    def test_property_22_ai_decision_stored(self):
        """
        Feature: ai-data-analyst-platform, Property 22: AI Decision Persistence

        For any AI recommendation generated, the decision should be stored in
        DynamoDB with session ID, reasoning, and confidence score.
        """
        self.table_mock.put_item.return_value = {}

        recommendations = {
            'recommended_models': [
                {'model': 'random_forest', 'confidence': 0.85, 'reasoning': 'Good for mixed data'}
            ],
            'analysis_type': 'supervised_classification',
            'reasoning': 'Test reasoning',
            'source': 'bedrock_ai'
        }
        characteristics = {'num_rows': 100, 'num_columns': 5}

        store_ai_decision("test-session", recommendations, characteristics)

        self.table_mock.put_item.assert_called_once()
        item = self.table_mock.put_item.call_args[1]['Item']

        assert item['session_id'] == 'test-session'
        assert item['decision_type'] == 'model_recommendation'
        assert 'timestamp' in item
        assert 'recommendation' in item
        assert item['reasoning'] == 'Test reasoning'
        assert item['confidence_score'] == Decimal('0.85')
        assert item['primary_model'] == 'random_forest'
        assert item['source'] == 'bedrock_ai'
        assert 'data_characteristics' in item

    @given(df=dataset_with_target())
    @settings(max_examples=10)
    def test_property_22_full_recommendation_persisted(self, df):
        """All generated recommendations should be persistable."""
        self.table_mock.reset_mock()
        self.table_mock.put_item.return_value = {}

        characteristics = analyze_dataset_characteristics(df)
        recommendations = generate_rule_based_recommendations(characteristics, None)

        store_ai_decision("session-123", recommendations, characteristics)

        self.table_mock.put_item.assert_called_once()
        item = self.table_mock.put_item.call_args[1]['Item']
        assert item['session_id'] == 'session-123'
        assert float(item['confidence_score']) >= 0.0
        assert float(item['confidence_score']) <= 1.0


class TestAIResponseCaching:
    """
    Property 26: AI Response Caching
    Validates: Requirements 15.1, 15.2, 15.3, 15.4
    """

    def setup_method(self):
        self.dynamodb_mock = Mock()
        self.table_mock = Mock()
        self.dynamodb_mock.Table.return_value = self.table_mock
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.dynamodb_patcher.stop()

    def test_property_26_prompt_hash_consistency(self):
        """
        Feature: ai-data-analyst-platform, Property 26: AI Response Caching

        Same prompts should produce the same hash consistently.
        """
        prompt = "Analyze this dataset with 100 rows and 5 columns"
        hash1 = generate_prompt_hash(prompt)
        hash2 = generate_prompt_hash(prompt)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_property_26_different_prompts_different_hashes(self):
        """Different prompts should produce different hashes."""
        hash1 = generate_prompt_hash("prompt A")
        hash2 = generate_prompt_hash("prompt B")
        assert hash1 != hash2

    def test_property_26_cache_hit(self):
        """Cached responses should be returned on cache hit."""
        cached_data = {
            'recommended_models': [{'model': 'rf', 'confidence': 0.8, 'reasoning': 'test'}],
            'analysis_type': 'supervised_classification'
        }

        self.table_mock.get_item.return_value = {
            'Item': {
                'prompt_hash': 'testhash',
                'response': json.dumps(cached_data),
                'timestamp': datetime.utcnow().isoformat(),
                'ttl': int((datetime.utcnow() + timedelta(hours=24)).timestamp())
            }
        }

        result = check_cache('testhash')
        assert result is not None
        assert result['recommended_models'][0]['model'] == 'rf'

    def test_property_26_cache_miss(self):
        """Empty cache should return None."""
        self.table_mock.get_item.return_value = {}

        result = check_cache('nonexistent_hash')
        assert result is None

    def test_property_26_cache_store(self):
        """Responses should be stored in cache with TTL."""
        self.table_mock.put_item.return_value = {}

        data = {'recommended_models': [], 'reasoning': 'test'}
        store_cache('test_hash', data)

        self.table_mock.put_item.assert_called_once()
        item = self.table_mock.put_item.call_args[1]['Item']
        assert item['prompt_hash'] == 'test_hash'
        assert 'response' in item
        assert 'timestamp' in item
        assert 'ttl' in item

    def test_property_26_expired_cache_not_returned(self):
        """Expired cache entries should not be returned."""
        expired_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()

        self.table_mock.get_item.return_value = {
            'Item': {
                'prompt_hash': 'expired_hash',
                'response': json.dumps({'models': []}),
                'timestamp': expired_time,
                'ttl': 0
            }
        }

        result = check_cache('expired_hash')
        assert result is None


class TestQualityBasedRecommendations:
    """
    Property 29: Quality-Based AI Recommendations
    Validates: Requirements 16.3
    """

    @given(report=quality_report_data())
    @settings(max_examples=25)
    def test_property_29_quality_recommendations_generated(self, report):
        """
        Feature: ai-data-analyst-platform, Property 29: Quality-Based AI Recommendations

        For any dataset with quality metrics, the AI Assistant should provide
        preprocessing recommendations based on the quality score.
        """
        recommendations = generate_quality_based_recommendations(report)

        # Should return a list
        assert isinstance(recommendations, list)

        # Each recommendation should have required fields
        for rec in recommendations:
            assert 'category' in rec
            assert 'action' in rec
            assert 'priority' in rec
            assert 'reasoning' in rec
            assert rec['priority'] in ['low', 'medium', 'high']

    def test_high_missing_triggers_recommendation(self):
        """High missing values should trigger high-priority recommendation."""
        report = {
            'overall_quality_score': 40,
            'missing_value_analysis': {'overall_missing_percentage': 25},
            'duplicate_analysis': {'duplicate_count': 0},
            'data_imbalance_analysis': {'max_imbalance_ratio': 1.0}
        }
        recs = generate_quality_based_recommendations(report)

        missing_recs = [r for r in recs if r['category'] == 'missing_values']
        assert len(missing_recs) > 0
        assert missing_recs[0]['priority'] == 'high'

    def test_duplicates_trigger_recommendation(self):
        """Duplicate rows should trigger a removal recommendation."""
        report = {
            'overall_quality_score': 80,
            'missing_value_analysis': {'overall_missing_percentage': 0},
            'duplicate_analysis': {'duplicate_count': 10},
            'data_imbalance_analysis': {'max_imbalance_ratio': 1.0}
        }
        recs = generate_quality_based_recommendations(report)

        dup_recs = [r for r in recs if r['category'] == 'duplicates']
        assert len(dup_recs) == 1
        assert dup_recs[0]['action'] == 'remove_duplicates'

    def test_imbalance_trigger_recommendation(self):
        """High imbalance ratio should trigger recommendation."""
        report = {
            'overall_quality_score': 70,
            'missing_value_analysis': {'overall_missing_percentage': 0},
            'duplicate_analysis': {'duplicate_count': 0},
            'data_imbalance_analysis': {'max_imbalance_ratio': 5.0}
        }
        recs = generate_quality_based_recommendations(report)

        imbalance_recs = [r for r in recs if r['category'] == 'class_imbalance']
        assert len(imbalance_recs) == 1
        assert imbalance_recs[0]['priority'] == 'high'

    def test_low_quality_score_triggers_overall_recommendation(self):
        """Low overall quality should trigger extensive preprocessing recommendation."""
        report = {
            'overall_quality_score': 30,
            'missing_value_analysis': {'overall_missing_percentage': 0},
            'duplicate_analysis': {'duplicate_count': 0},
            'data_imbalance_analysis': {'max_imbalance_ratio': 1.0}
        }
        recs = generate_quality_based_recommendations(report)

        overall_recs = [r for r in recs if r['category'] == 'overall']
        assert len(overall_recs) == 1
        assert overall_recs[0]['action'] == 'extensive_preprocessing'

    def test_clean_dataset_minimal_recommendations(self):
        """Clean datasets should generate minimal recommendations."""
        report = {
            'overall_quality_score': 95,
            'missing_value_analysis': {'overall_missing_percentage': 0},
            'duplicate_analysis': {'duplicate_count': 0},
            'data_imbalance_analysis': {'max_imbalance_ratio': 1.2}
        }
        recs = generate_quality_based_recommendations(report)
        # Clean dataset should have no or very few recommendations
        assert len(recs) <= 1


class TestBedrockRetryLogic:
    """Tests for Bedrock retry logic (supports Property 32)."""

    def setup_method(self):
        self.bedrock_mock = Mock()
        self.bedrock_patcher = patch.object(_module, 'bedrock_client', self.bedrock_mock)
        self.time_patcher = patch.object(_module.time, 'sleep')
        self.bedrock_patcher.start()
        self.time_patcher.start()

    def teardown_method(self):
        self.bedrock_patcher.stop()
        self.time_patcher.stop()

    def test_successful_first_attempt(self):
        """Successful first attempt should return immediately."""
        self.bedrock_mock.invoke_model.return_value = {
            'body': Mock(read=Mock(return_value=json.dumps({
                'content': [{'text': '{"models": []}'}]
            }).encode()))
        }

        result = invoke_bedrock_with_retry("test prompt")
        assert result == '{"models": []}'
        assert self.bedrock_mock.invoke_model.call_count == 1

    def test_retry_on_failure_then_success(self):
        """Should retry on failure and succeed on second attempt."""
        self.bedrock_mock.invoke_model.side_effect = [
            Exception("ThrottlingException"),
            {
                'body': Mock(read=Mock(return_value=json.dumps({
                    'content': [{'text': 'success'}]
                }).encode()))
            }
        ]

        result = invoke_bedrock_with_retry("test prompt")
        assert result == 'success'
        assert self.bedrock_mock.invoke_model.call_count == 2

    def test_all_retries_exhausted(self):
        """Should return None after all retries are exhausted."""
        self.bedrock_mock.invoke_model.side_effect = Exception("ServiceUnavailable")

        result = invoke_bedrock_with_retry("test prompt")
        assert result is None
        assert self.bedrock_mock.invoke_model.call_count == 3


class TestAIResponseParsing:
    """Tests for parsing AI responses."""

    def test_valid_json_response(self):
        """Valid JSON response should be parsed correctly."""
        ai_text = '{"recommended_models": [{"model": "random_forest", "confidence": 0.85, "reasoning": "good"}], "analysis_type": "supervised_classification", "reasoning": "test"}'
        result = parse_ai_response(ai_text, {})

        assert result['source'] == 'bedrock_ai'
        assert len(result['recommended_models']) == 1
        assert result['recommended_models'][0]['model'] == 'random_forest'
        assert result['recommended_models'][0]['confidence'] == 0.85

    def test_json_embedded_in_text(self):
        """JSON embedded in text should be extracted."""
        ai_text = 'Here is my analysis:\n{"recommended_models": [], "analysis_type": "exploratory", "reasoning": "test"}\nEnd.'
        result = parse_ai_response(ai_text, {})

        assert result['source'] == 'bedrock_ai'
        assert result['analysis_type'] == 'exploratory'

    def test_invalid_json_fallback(self):
        """Invalid JSON should fall back to raw response."""
        ai_text = "This is not valid JSON at all"
        result = parse_ai_response(ai_text, {})

        assert result['source'] == 'bedrock_ai_raw'
        assert result['reasoning'] == ai_text

    def test_confidence_clamping(self):
        """Confidence scores should be clamped to [0, 1]."""
        ai_text = '{"recommended_models": [{"model": "rf", "confidence": 1.5, "reasoning": "test"}], "analysis_type": "exploratory", "reasoning": "test"}'
        result = parse_ai_response(ai_text, {})

        assert result['recommended_models'][0]['confidence'] == 1.0


class TestLambdaHandlerIntegration:
    """Integration tests for the full lambda handler."""

    def setup_method(self):
        self.s3_mock = Mock()
        self.dynamodb_mock = Mock()
        self.bedrock_mock = Mock()
        self.sessions_table = Mock()
        self.cache_table = Mock()
        self.ai_decisions_table = Mock()
        self.operations_table = Mock()

        def table_factory(name):
            if 'sessions' in name:
                return self.sessions_table
            elif 'cache' in name:
                return self.cache_table
            elif 'ai-decisions' in name or 'decisions' in name:
                return self.ai_decisions_table
            elif 'operations' in name:
                return self.operations_table
            return Mock()

        self.dynamodb_mock.Table = table_factory

        self.s3_patcher = patch.object(_module, 's3_client', self.s3_mock)
        self.dynamodb_patcher = patch.object(_module, 'dynamodb', self.dynamodb_mock)
        self.bedrock_patcher = patch.object(_module, 'bedrock_client', self.bedrock_mock)
        self.time_patcher = patch.object(_module.time, 'sleep')

        self.s3_patcher.start()
        self.dynamodb_patcher.start()
        self.bedrock_patcher.start()
        self.time_patcher.start()

    def teardown_method(self):
        self.s3_patcher.stop()
        self.dynamodb_patcher.stop()
        self.bedrock_patcher.stop()
        self.time_patcher.stop()

    def test_missing_session_id(self):
        """Should return 400 for missing session ID."""
        event = {'pathParameters': {}}
        response = lambda_handler(event, {})
        assert response['statusCode'] == 400

    def test_session_not_found(self):
        """Should return 404 for non-existent session."""
        self.sessions_table.get_item.return_value = {}

        event = {'pathParameters': {'sessionId': 'nonexistent'}}
        response = lambda_handler(event, {})
        assert response['statusCode'] == 404

    def test_full_recommendation_flow_with_fallback(self):
        """Full flow should work with rule-based fallback when Bedrock fails."""
        session_id = "test-session"
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f2': [5.0, 4.0, 3.0, 2.0, 1.0],
            'target': ['A', 'B', 'A', 'B', 'A']
        })

        self.sessions_table.get_item.return_value = {
            'Item': {'session_id': session_id, 'status': 'uploaded'}
        }

        csv_bytes = df.to_csv(index=False).encode()
        self.s3_mock.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=csv_bytes))
        }

        # Cache miss
        self.cache_table.get_item.return_value = {}
        self.cache_table.put_item.return_value = {}
        self.ai_decisions_table.put_item.return_value = {}
        self.operations_table.put_item.return_value = {}
        self.operations_table.query.return_value = {'Items': []}

        # Bedrock fails - triggers fallback
        self.bedrock_mock.invoke_model.side_effect = Exception("Service unavailable")

        event = {'pathParameters': {'sessionId': session_id}}
        response = lambda_handler(event, {})

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['session_id'] == session_id
        assert 'recommendations' in body
        assert 'data_characteristics' in body
        assert len(body['recommendations']['recommended_models']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
