import json
import boto3
import boto3.dynamodb.conditions
import hashlib
import logging
import os
import io
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

s3_client = boto3.client('s3', region_name=aws_region)
dynamodb = boto3.resource('dynamodb', region_name=aws_region)
bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'ai-data-analyst-platform-data-dev-077437903006')
SESSIONS_TABLE = os.environ.get('SESSIONS_TABLE', 'ai-data-analyst-platform-sessions-dev')
AI_DECISIONS_TABLE = os.environ.get('AI_DECISIONS_TABLE', 'ai-data-analyst-platform-ai-decisions-dev')
CACHE_TABLE = os.environ.get('CACHE_TABLE', 'ai-data-analyst-platform-cache-dev')
OPERATIONS_TABLE = os.environ.get('OPERATIONS_TABLE', 'ai-data-analyst-platform-operations-dev')

# Bedrock configuration
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # exponential backoff delays in seconds
CACHE_TTL_HOURS = 24


def lambda_handler(event, context):
    """
    Main Lambda handler for AI Assistant recommendations.

    Supports:
    - GET /recommendations/{sessionId}: Generate AI recommendations for a dataset
    """
    try:
        # Parse request
        http_method = event.get('httpMethod', 'GET')
        path_params = event.get('pathParameters', {}) or {}
        session_id = path_params.get('sessionId')
        query_params = event.get('queryStringParameters', {}) or {}

        if not session_id:
            return create_response(400, {'error': 'Missing sessionId path parameter'})

        logger.info(f"AI Assistant invoked for session {session_id}")

        # Load session metadata
        session_data = load_session_data(session_id)
        if not session_data:
            return create_response(404, {'error': f'Session {session_id} not found'})

        # Load dataset
        dataset = load_dataset(session_id)
        if dataset is None:
            return create_response(404, {'error': 'Dataset not found for session'})

        # Analyze dataset characteristics
        characteristics = analyze_dataset_characteristics(dataset)

        # Load quality assessment if available
        quality_report = load_quality_report(session_id)

        # Build prompt for Bedrock
        prompt = build_recommendation_prompt(characteristics, quality_report)

        # Check cache first
        prompt_hash = generate_prompt_hash(prompt)
        cached_response = check_cache(prompt_hash)

        if cached_response:
            logger.info(f"Cache HIT for session {session_id}, hash: {prompt_hash}")
            recommendations = cached_response
        else:
            logger.info(f"Cache MISS for session {session_id}, hash: {prompt_hash}")
            # Call Bedrock with retry logic
            ai_response = invoke_bedrock_with_retry(prompt)

            if ai_response:
                recommendations = parse_ai_response(ai_response, characteristics)
            else:
                # Fallback to rule-based recommendations
                logger.warning("Bedrock failed, using rule-based fallback")
                recommendations = generate_rule_based_recommendations(characteristics, quality_report)

            # Cache the response
            store_cache(prompt_hash, recommendations)

        # Add quality-based recommendations if available
        if quality_report:
            quality_recommendations = generate_quality_based_recommendations(quality_report)
            recommendations['quality_recommendations'] = quality_recommendations
            recommendations['quality_score'] = quality_report.get('overall_quality_score', 0)

        # Store AI decision in DynamoDB
        store_ai_decision(session_id, recommendations, characteristics)

        # Log the operation
        log_operation(session_id, 'ai_recommendation', 'completed')

        return create_response(200, {
            'session_id': session_id,
            'recommendations': recommendations,
            'data_characteristics': characteristics,
            'cached': cached_response is not None
        })

    except Exception as e:
        logger.error(f"Error in AI Assistant: {str(e)}")
        return create_response(500, {'error': f'Internal server error: {str(e)}'})


def load_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session metadata from DynamoDB."""
    try:
        table = dynamodb.Table(SESSIONS_TABLE)
        response = table.get_item(Key={'session_id': session_id})
        return response.get('Item')
    except Exception as e:
        logger.error(f"Error loading session data: {str(e)}")
        return None


def load_dataset(session_id: str) -> Optional[pd.DataFrame]:
    """Load dataset from S3 (prefer processed, fall back to original)."""
    for dataset_type in ['processed', 'original']:
        try:
            key = f"datasets/{session_id}/{dataset_type}.csv"
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            csv_content = response['Body'].read().decode('utf-8')
            return pd.read_csv(io.StringIO(csv_content))
        except Exception:
            continue
    logger.error(f"No dataset found for session {session_id}")
    return None


def load_quality_report(session_id: str) -> Optional[Dict[str, Any]]:
    """Load quality assessment report from DynamoDB operations table."""
    try:
        table = dynamodb.Table(OPERATIONS_TABLE)
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(session_id),
            ScanIndexForward=False,
            Limit=20
        )
        for item in response.get('Items', []):
            if item.get('operation_type') == 'quality_assessment':
                return item.get('quality_report', {})
        return None
    except Exception as e:
        logger.error(f"Error loading quality report: {str(e)}")
        return None


def analyze_dataset_characteristics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset to determine key characteristics for AI recommendations."""
    num_rows, num_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Missing value analysis
    total_missing = int(df.isnull().sum().sum())
    missing_percentage = round((total_missing / (num_rows * num_cols)) * 100, 2) if num_rows * num_cols > 0 else 0

    # Size classification
    if num_rows < 100:
        size_class = 'small'
    elif num_rows < 10000:
        size_class = 'medium'
    else:
        size_class = 'large'

    # Detect potential target column (last column heuristic + unique value analysis)
    potential_targets = []
    for col in df.columns:
        nunique = df[col].nunique()
        if 2 <= nunique <= 20 and nunique < num_rows * 0.1:
            potential_targets.append({
                'column': col,
                'unique_values': nunique,
                'type': 'classification'
            })

    # Check for regression targets (numeric columns that could be targets)
    for col in numeric_cols:
        nunique = df[col].nunique()
        if nunique > 20 and nunique > num_rows * 0.5:
            potential_targets.append({
                'column': col,
                'unique_values': nunique,
                'type': 'regression'
            })

    # Detect high cardinality columns
    high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]

    # Check correlation among numeric features
    has_high_correlation = False
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            has_high_correlation = (upper > 0.8).any().any()
        except Exception:
            pass

    characteristics = {
        'num_rows': num_rows,
        'num_columns': num_cols,
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'boolean_columns': len(bool_cols),
        'datetime_columns': len(datetime_cols),
        'numeric_column_names': numeric_cols,
        'categorical_column_names': categorical_cols,
        'missing_values': total_missing,
        'missing_percentage': missing_percentage,
        'size_class': size_class,
        'potential_targets': potential_targets,
        'high_cardinality_columns': high_cardinality,
        'has_high_correlation': has_high_correlation,
        'has_missing_values': total_missing > 0,
        'has_categorical': len(categorical_cols) > 0,
    }

    return characteristics


def build_recommendation_prompt(characteristics: Dict[str, Any],
                                quality_report: Optional[Dict[str, Any]]) -> str:
    """Build the prompt for Bedrock AI model."""
    prompt = f"""You are a data science expert. Analyze the following dataset characteristics and provide machine learning recommendations.

Dataset Characteristics:
- Rows: {characteristics['num_rows']}
- Columns: {characteristics['num_columns']}
- Numeric columns: {characteristics['numeric_columns']}
- Categorical columns: {characteristics['categorical_columns']}
- Missing values: {characteristics['missing_percentage']}%
- Size class: {characteristics['size_class']}
- Has high correlation: {characteristics['has_high_correlation']}
- Has categorical features: {characteristics['has_categorical']}
- Potential target columns: {json.dumps(characteristics['potential_targets'])}
"""

    if quality_report:
        quality_score = quality_report.get('overall_quality_score', 'N/A')
        prompt += f"\nDataset Quality Score: {quality_score}/100\n"

    prompt += """
Respond in JSON format with the following structure:
{
  "recommended_models": [
    {"model": "model_name", "confidence": 0.0-1.0, "reasoning": "why this model"}
  ],
  "recommended_preprocessing": [
    {"step": "step_name", "reasoning": "why this step"}
  ],
  "analysis_type": "supervised_classification|supervised_regression|unsupervised_clustering|exploratory",
  "reasoning": "overall analysis reasoning"
}
"""
    return prompt


def generate_prompt_hash(prompt: str) -> str:
    """Generate SHA-256 hash of the prompt for caching."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def check_cache(prompt_hash: str) -> Optional[Dict[str, Any]]:
    """Check DynamoDB cache for existing response."""
    try:
        table = dynamodb.Table(CACHE_TABLE)
        response = table.get_item(Key={'prompt_hash': prompt_hash})
        item = response.get('Item')
        if item:
            # Check if the cached response is still valid (TTL may not have cleaned it up yet)
            cached_time = item.get('timestamp', '')
            if cached_time:
                cached_dt = datetime.fromisoformat(cached_time)
                if datetime.utcnow() - cached_dt < timedelta(hours=CACHE_TTL_HOURS):
                    logger.info(f"Cache hit for hash {prompt_hash[:16]}...")
                    return json.loads(item['response']) if isinstance(item['response'], str) else item['response']
            logger.info(f"Cache entry expired for hash {prompt_hash[:16]}...")
        return None
    except Exception as e:
        logger.error(f"Error checking cache: {str(e)}")
        return None


def store_cache(prompt_hash: str, response: Dict[str, Any]) -> None:
    """Store AI response in DynamoDB cache."""
    try:
        table = dynamodb.Table(CACHE_TABLE)
        ttl_seconds = int((datetime.utcnow() + timedelta(hours=CACHE_TTL_HOURS)).timestamp())
        table.put_item(Item={
            'prompt_hash': prompt_hash,
            'response': json.dumps(response, default=str),
            'timestamp': datetime.utcnow().isoformat(),
            'ttl': ttl_seconds
        })
        logger.info(f"Cached response for hash {prompt_hash[:16]}...")
    except Exception as e:
        logger.error(f"Error storing cache: {str(e)}")


def invoke_bedrock_with_retry(prompt: str) -> Optional[str]:
    """
    Invoke Amazon Bedrock with exponential backoff retry logic.

    Retries up to MAX_RETRIES times with delays of 1s, 2s, 4s.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })

            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=request_body
            )

            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            if content and len(content) > 0:
                return content[0].get('text', '')

            return None

        except Exception as e:
            last_error = e
            logger.warning(f"Bedrock attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)

    logger.error(f"All Bedrock retry attempts failed. Last error: {str(last_error)}")
    return None


def parse_ai_response(ai_text: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Parse AI response text into structured recommendations."""
    try:
        # Try to extract JSON from the response
        json_start = ai_text.find('{')
        json_end = ai_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = ai_text[json_start:json_end]
            parsed = json.loads(json_str)

            # Validate and normalize the response
            recommendations = {
                'recommended_models': parsed.get('recommended_models', []),
                'recommended_preprocessing': parsed.get('recommended_preprocessing', []),
                'analysis_type': parsed.get('analysis_type', 'exploratory'),
                'reasoning': parsed.get('reasoning', ''),
                'source': 'bedrock_ai',
                'model_id': BEDROCK_MODEL_ID,
            }

            # Ensure confidence scores are present and valid
            for model in recommendations['recommended_models']:
                if 'confidence' not in model:
                    model['confidence'] = 0.5
                model['confidence'] = max(0.0, min(1.0, float(model['confidence'])))

            return recommendations

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse AI response as JSON: {str(e)}")

    # If JSON parsing fails, return the raw response with structure
    return {
        'recommended_models': [],
        'recommended_preprocessing': [],
        'analysis_type': 'exploratory',
        'reasoning': ai_text,
        'source': 'bedrock_ai_raw',
        'model_id': BEDROCK_MODEL_ID,
    }


def generate_rule_based_recommendations(characteristics: Dict[str, Any],
                                        quality_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate rule-based recommendations as a fallback when Bedrock is unavailable.
    Uses dataset characteristics to suggest appropriate models and preprocessing.
    """
    models = []
    preprocessing = []
    analysis_type = 'exploratory'

    # Determine analysis type from potential targets
    targets = characteristics.get('potential_targets', [])
    if targets:
        primary_target = targets[0]
        if primary_target['type'] == 'classification':
            analysis_type = 'supervised_classification'

            if characteristics['num_rows'] < 1000:
                models.append({
                    'model': 'logistic_regression',
                    'confidence': 0.8,
                    'reasoning': 'Good baseline for small classification datasets'
                })
                models.append({
                    'model': 'knn',
                    'confidence': 0.65,
                    'reasoning': 'Works well with small to medium datasets'
                })
            else:
                models.append({
                    'model': 'random_forest',
                    'confidence': 0.85,
                    'reasoning': 'Robust ensemble method for medium to large datasets with mixed features'
                })
                models.append({
                    'model': 'svm',
                    'confidence': 0.7,
                    'reasoning': 'Effective for classification with clear margins'
                })

            models.append({
                'model': 'random_forest',
                'confidence': 0.75,
                'reasoning': 'Provides feature importance and handles mixed data types'
            })

        elif primary_target['type'] == 'regression':
            analysis_type = 'supervised_regression'
            models.append({
                'model': 'random_forest',
                'confidence': 0.8,
                'reasoning': 'Robust for regression tasks with non-linear relationships'
            })
    else:
        # No clear target - suggest unsupervised
        analysis_type = 'unsupervised_clustering'
        models.append({
            'model': 'kmeans',
            'confidence': 0.75,
            'reasoning': 'Standard clustering for discovering groups in the data'
        })
        models.append({
            'model': 'dbscan',
            'confidence': 0.6,
            'reasoning': 'Density-based clustering that can find arbitrary-shaped clusters'
        })

    # Preprocessing recommendations
    if characteristics['has_missing_values']:
        if characteristics['missing_percentage'] < 5:
            preprocessing.append({
                'step': 'fill_null_mean',
                'reasoning': f"Low missing value percentage ({characteristics['missing_percentage']}%), mean imputation suitable"
            })
        elif characteristics['missing_percentage'] < 20:
            preprocessing.append({
                'step': 'fill_null_median',
                'reasoning': f"Moderate missing values ({characteristics['missing_percentage']}%), median imputation is more robust"
            })
        else:
            preprocessing.append({
                'step': 'remove_null',
                'reasoning': f"High missing value percentage ({characteristics['missing_percentage']}%), consider removing rows"
            })

    if characteristics['has_categorical']:
        if characteristics.get('high_cardinality_columns'):
            preprocessing.append({
                'step': 'label_encoding',
                'reasoning': 'High cardinality categorical columns found - label encoding recommended to avoid dimensionality explosion'
            })
        else:
            preprocessing.append({
                'step': 'one_hot_encoding',
                'reasoning': 'Low-cardinality categorical columns - one-hot encoding preserves information'
            })

    if characteristics['numeric_columns'] > 0 and analysis_type != 'unsupervised_clustering':
        preprocessing.append({
            'step': 'standard_scaling',
            'reasoning': 'Scaling numeric features for better model performance'
        })

    if characteristics['has_high_correlation']:
        preprocessing.append({
            'step': 'outlier_removal_iqr',
            'reasoning': 'High correlations detected - removing outliers may improve model accuracy'
        })

    # Deduplicate models by name
    seen_models = set()
    unique_models = []
    for m in models:
        if m['model'] not in seen_models:
            seen_models.add(m['model'])
            unique_models.append(m)

    return {
        'recommended_models': unique_models,
        'recommended_preprocessing': preprocessing,
        'analysis_type': analysis_type,
        'reasoning': f"Rule-based analysis for {characteristics['size_class']} "
                     f"dataset with {characteristics['num_rows']} rows and "
                     f"{characteristics['num_columns']} columns.",
        'source': 'rule_based_fallback',
    }


def generate_quality_based_recommendations(quality_report: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate preprocessing recommendations based on dataset quality scores.
    Implements Requirement 16.3.
    """
    recommendations = []
    quality_score = quality_report.get('overall_quality_score', 100)

    # Missing value recommendations
    missing_analysis = quality_report.get('missing_value_analysis', {})
    overall_missing_pct = missing_analysis.get('overall_missing_percentage', 0)
    if overall_missing_pct > 0:
        if overall_missing_pct < 5:
            recommendations.append({
                'category': 'missing_values',
                'action': 'fill_null',
                'priority': 'low',
                'reasoning': f'Minor missing values ({overall_missing_pct}%) - simple imputation recommended'
            })
        elif overall_missing_pct < 20:
            recommendations.append({
                'category': 'missing_values',
                'action': 'fill_null_median',
                'priority': 'medium',
                'reasoning': f'Moderate missing values ({overall_missing_pct}%) - median imputation recommended'
            })
        else:
            recommendations.append({
                'category': 'missing_values',
                'action': 'review_columns',
                'priority': 'high',
                'reasoning': f'Significant missing values ({overall_missing_pct}%) - review columns with high missing rates'
            })

    # Duplicate recommendations
    duplicate_analysis = quality_report.get('duplicate_analysis', {})
    dup_count = duplicate_analysis.get('duplicate_count', 0)
    if dup_count > 0:
        recommendations.append({
            'category': 'duplicates',
            'action': 'remove_duplicates',
            'priority': 'medium',
            'reasoning': f'{dup_count} duplicate rows detected - removal recommended for better model performance'
        })

    # Imbalance recommendations
    imbalance_analysis = quality_report.get('data_imbalance_analysis', {})
    imbalance_ratio = imbalance_analysis.get('max_imbalance_ratio', 1.0)
    if isinstance(imbalance_ratio, (int, float)) and imbalance_ratio > 3.0:
        recommendations.append({
            'category': 'class_imbalance',
            'action': 'consider_sampling',
            'priority': 'high',
            'reasoning': f'Class imbalance detected (ratio: {imbalance_ratio:.1f}) - consider oversampling minority class or using balanced models'
        })

    # Overall quality recommendations
    if quality_score < 50:
        recommendations.append({
            'category': 'overall',
            'action': 'extensive_preprocessing',
            'priority': 'high',
            'reasoning': f'Low quality score ({quality_score}/100) - extensive data cleaning recommended before modeling'
        })
    elif quality_score < 75:
        recommendations.append({
            'category': 'overall',
            'action': 'moderate_preprocessing',
            'priority': 'medium',
            'reasoning': f'Moderate quality score ({quality_score}/100) - some preprocessing recommended'
        })

    return recommendations


def store_ai_decision(session_id: str, recommendations: Dict[str, Any],
                      characteristics: Dict[str, Any]) -> None:
    """Store AI decision and reasoning in DynamoDB AI Decisions table."""
    try:
        table = dynamodb.Table(AI_DECISIONS_TABLE)

        # Extract primary recommendation info
        primary_model = ''
        confidence = 0.0
        models_list = recommendations.get('recommended_models', [])
        if models_list:
            primary_model = models_list[0].get('model', '')
            confidence = float(models_list[0].get('confidence', 0))

        item = {
            'session_id': session_id,
            'decision_type': 'model_recommendation',
            'timestamp': datetime.utcnow().isoformat(),
            'recommendation': json.dumps(recommendations, default=str),
            'reasoning': recommendations.get('reasoning', ''),
            'confidence_score': Decimal(str(round(confidence, 4))),
            'analysis_type': recommendations.get('analysis_type', 'exploratory'),
            'primary_model': primary_model,
            'source': recommendations.get('source', 'unknown'),
            'data_characteristics': json.dumps(characteristics, default=str),
            'ttl': int((datetime.utcnow() + timedelta(days=30)).timestamp())
        }

        table.put_item(Item=item)
        logger.info(f"AI decision stored for session {session_id}")

    except Exception as e:
        logger.error(f"Error storing AI decision: {str(e)}")


def log_operation(session_id: str, operation_type: str, status: str) -> None:
    """Log operation to DynamoDB Operations table."""
    try:
        table = dynamodb.Table(OPERATIONS_TABLE)
        table.put_item(Item={
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'operation_type': operation_type,
            'status': status
        })
    except Exception as e:
        logger.error(f"Error logging operation: {str(e)}")


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create API Gateway response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,OPTIONS'
        },
        'body': json.dumps(body, default=str)
    }
