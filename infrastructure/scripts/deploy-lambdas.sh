#!/bin/bash

# Lambda Functions Deployment Script
# Deploys all Lambda functions for AI Data Analyst Platform

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

PROJECT_NAME="ai-data-analyst-platform"
ENVIRONMENT="${1:-dev}"
AWS_REGION="${2:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== Deploying Lambda Functions ==="
echo "Project: $PROJECT_NAME"
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"
echo "Account: $ACCOUNT_ID"

# Lambda function configurations
# Format: dir_name:function_name:memory:timeout
FUNCTIONS=(
    "upload:${PROJECT_NAME}-upload-${ENVIRONMENT}:1024:300"
    "preview:${PROJECT_NAME}-preview-${ENVIRONMENT}:1024:300"
    "processing:${PROJECT_NAME}-processing-${ENVIRONMENT}:2048:600"
    "ml_training:${PROJECT_NAME}-ml-training-${ENVIRONMENT}:2048:900"
    "visualization:${PROJECT_NAME}-visualization-${ENVIRONMENT}:1024:300"
    "ai_assistant:${PROJECT_NAME}-ai-assistant-${ENVIRONMENT}:1024:300"
)

# Create IAM role for Lambda functions if it doesn't exist
ROLE_NAME="${PROJECT_NAME}-lambda-role-${ENVIRONMENT}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

create_lambda_role() {
    echo ""
    echo "--- Creating IAM Role ---"
    
    if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
        echo "Role $ROLE_NAME already exists"
        ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
    else
        # Create the role
        aws iam create-role \
            --role-name "$ROLE_NAME" \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' \
            --tags Key=Project,Value=$PROJECT_NAME Key=Environment,Value=$ENVIRONMENT \
            --output text --query 'Role.Arn'

        ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
        echo "Created role: $ROLE_ARN"

        # Attach basic Lambda execution policy
        aws iam attach-role-policy \
            --role-name "$ROLE_NAME" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

        # Create and attach custom policy for S3, DynamoDB, Bedrock access
        POLICY_NAME="${PROJECT_NAME}-lambda-policy-${ENVIRONMENT}"
        
        aws iam put-role-policy \
            --role-name "$ROLE_NAME" \
            --policy-name "$POLICY_NAME" \
            --policy-document "{
                \"Version\": \"2012-10-17\",
                \"Statement\": [
                    {
                        \"Effect\": \"Allow\",
                        \"Action\": [
                            \"s3:GetObject\",
                            \"s3:PutObject\",
                            \"s3:ListBucket\",
                            \"s3:DeleteObject\"
                        ],
                        \"Resource\": [
                            \"arn:aws:s3:::${PROJECT_NAME}-data-${ENVIRONMENT}-${ACCOUNT_ID}\",
                            \"arn:aws:s3:::${PROJECT_NAME}-data-${ENVIRONMENT}-${ACCOUNT_ID}/*\"
                        ]
                    },
                    {
                        \"Effect\": \"Allow\",
                        \"Action\": [
                            \"dynamodb:GetItem\",
                            \"dynamodb:PutItem\",
                            \"dynamodb:UpdateItem\",
                            \"dynamodb:Query\",
                            \"dynamodb:Scan\"
                        ],
                        \"Resource\": [
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-sessions-${ENVIRONMENT}\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-sessions-${ENVIRONMENT}/*\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-operations-${ENVIRONMENT}\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-operations-${ENVIRONMENT}/*\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-ai-decisions-${ENVIRONMENT}\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-ai-decisions-${ENVIRONMENT}/*\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-cache-${ENVIRONMENT}\",
                            \"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${PROJECT_NAME}-cache-${ENVIRONMENT}/*\"
                        ]
                    },
                    {
                        \"Effect\": \"Allow\",
                        \"Action\": [
                            \"bedrock:InvokeModel\",
                            \"bedrock:ApplyGuardrail\"
                        ],
                        \"Resource\": \"*\"
                    }
                ]
            }"

        echo "Attached policies. Waiting 10s for role propagation..."
        sleep 10
    fi

    echo "Using role: $ROLE_ARN"
}

# Package and deploy a single Lambda function
deploy_function() {
    local dir_name=$1
    local func_name=$2
    local memory=$3
    local timeout=$4
    
    echo ""
    echo "--- Deploying: $func_name (from lambda/$dir_name) ---"
    
    # Create temp directory for packaging
    local build_dir="$(pwd)/build-${dir_name}"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    
    # Copy lambda code
    cp lambda/${dir_name}/*.py "$build_dir/"
    
    # Copy shared utilities
    if [ -d "lambda/shared" ]; then
        mkdir -p "$build_dir/shared"
        cp lambda/shared/*.py "$build_dir/shared/"
    fi
    
    # Install dependencies for Lambda (Linux x86_64)
    if [ -f "lambda/${dir_name}/requirements.txt" ]; then
        echo "Installing dependencies..."
        pip install --no-user -r "lambda/${dir_name}/requirements.txt" -t "$build_dir" \
            --platform manylinux2014_x86_64 --implementation cp --python-version 3.11 \
            --only-binary=:all: --upgrade --quiet 2>/dev/null || \
        pip install --no-user -r "lambda/${dir_name}/requirements.txt" -t "$build_dir" --quiet 2>/dev/null
    fi
    
    # Create ZIP using Python (cross-platform)
    local zip_file="$(pwd)/${func_name}.zip"
    rm -f "$zip_file"
    python -c "
import zipfile, os
build_dir = '$build_dir'
zip_path = '$zip_file'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(build_dir):
        for f in files:
            full = os.path.join(root, f)
            arc = os.path.relpath(full, build_dir)
            zf.write(full, arc)
print(f'Created {zip_path}')
"
    
    local zip_size=$(wc -c < "$zip_file")
    echo "Package size: $(( zip_size / 1024 / 1024 ))MB"
    
    # Set environment variables based on function
    local env_vars="Variables={AWS_DEFAULT_REGION=$AWS_REGION"
    env_vars+=",DATA_BUCKET=${PROJECT_NAME}-data-${ENVIRONMENT}-${ACCOUNT_ID}"
    env_vars+=",BUCKET_NAME=${PROJECT_NAME}-data-${ENVIRONMENT}-${ACCOUNT_ID}"
    env_vars+=",SESSIONS_TABLE=${PROJECT_NAME}-sessions-${ENVIRONMENT}"
    env_vars+=",OPERATIONS_TABLE=${PROJECT_NAME}-operations-${ENVIRONMENT}"
    env_vars+=",AI_DECISIONS_TABLE=${PROJECT_NAME}-ai-decisions-${ENVIRONMENT}"
    env_vars+=",CACHE_TABLE=${PROJECT_NAME}-cache-${ENVIRONMENT}"
    
    if [ "$dir_name" == "ai_assistant" ]; then
        env_vars+=",BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0"
    fi
    env_vars+="}"

    # Check if function exists
    if aws lambda get-function --function-name "$func_name" --region "$AWS_REGION" &>/dev/null; then
        echo "Updating existing function..."
        aws lambda update-function-code \
            --function-name "$func_name" \
            --zip-file "fileb://$zip_file" \
            --region "$AWS_REGION" \
            --output text --query 'FunctionArn'
        
        # Wait for update to finish
        aws lambda wait function-updated --function-name "$func_name" --region "$AWS_REGION" 2>/dev/null || true
        
        aws lambda update-function-configuration \
            --function-name "$func_name" \
            --memory-size "$memory" \
            --timeout "$timeout" \
            --environment "$env_vars" \
            --region "$AWS_REGION" \
            --output text --query 'FunctionArn'
    else
        echo "Creating new function..."
        
        # If zip is > 50MB, upload to S3 first
        if [ "$zip_size" -gt 52428800 ]; then
            local s3_key="lambda-packages/${func_name}.zip"
            local s3_bucket="${PROJECT_NAME}-data-${ENVIRONMENT}-${ACCOUNT_ID}"
            echo "Package too large for direct upload, using S3..."
            aws s3 cp "$zip_file" "s3://$s3_bucket/$s3_key" --region "$AWS_REGION"
            
            aws lambda create-function \
                --function-name "$func_name" \
                --runtime python3.11 \
                --role "$ROLE_ARN" \
                --handler lambda_function.lambda_handler \
                --memory-size "$memory" \
                --timeout "$timeout" \
                --code "S3Bucket=$s3_bucket,S3Key=$s3_key" \
                --environment "$env_vars" \
                --region "$AWS_REGION" \
                --tags Project=$PROJECT_NAME,Environment=$ENVIRONMENT \
                --output text --query 'FunctionArn'
        else
            aws lambda create-function \
                --function-name "$func_name" \
                --runtime python3.11 \
                --role "$ROLE_ARN" \
                --handler lambda_function.lambda_handler \
                --memory-size "$memory" \
                --timeout "$timeout" \
                --zip-file "fileb://$zip_file" \
                --environment "$env_vars" \
                --region "$AWS_REGION" \
                --tags Project=$PROJECT_NAME,Environment=$ENVIRONMENT \
                --output text --query 'FunctionArn'
        fi
    fi

    # Wait for function to be active
    echo "Waiting for function to be active..."
    aws lambda wait function-active-v2 --function-name "$func_name" --region "$AWS_REGION" 2>/dev/null || true
    
    # Clean up
    rm -rf "$build_dir"
    rm -f "$zip_file"
    
    echo "Deployed: $func_name"
}

# Main execution
create_lambda_role

for func_config in "${FUNCTIONS[@]}"; do
    IFS=':' read -r dir_name func_name memory timeout <<< "$func_config"
    deploy_function "$dir_name" "$func_name" "$memory" "$timeout"
done

echo ""
echo "=== All Lambda functions deployed ==="
echo ""
echo "Lambda ARNs:"
for func_config in "${FUNCTIONS[@]}"; do
    IFS=':' read -r dir_name func_name memory timeout <<< "$func_config"
    arn=$(aws lambda get-function --function-name "$func_name" --region "$AWS_REGION" --query 'Configuration.FunctionArn' --output text 2>/dev/null)
    echo "  $dir_name: $arn"
done
