#!/bin/bash

# AI Data Analyst Platform Infrastructure Deployment Script
# This script deploys the complete AWS infrastructure using CloudFormation

set -e

# Navigate to project root (parent of infrastructure/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Configuration
PROJECT_NAME="ai-data-analyst-platform"
ENVIRONMENT="${1:-dev}"
AWS_REGION="${2:-us-east-1}"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if AWS CLI is installed and configured
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi

    print_success "AWS CLI is configured"
}

# Function to validate CloudFormation templates
validate_templates() {
    print_status "Validating CloudFormation templates..."
    
    local templates=(
        "s3-buckets.yaml"
        "dynamodb-tables.yaml"
        "api-gateway.yaml"
        "cloudfront.yaml"
        "master-template.yaml"
    )

    for template in "${templates[@]}"; do
        if aws cloudformation validate-template --template-body file://infrastructure/cloudformation/$template &> /dev/null; then
            print_success "Template $template is valid"
        else
            print_error "Template $template is invalid"
            exit 1
        fi
    done
}

# Function to upload templates to S3 (for nested stacks)
upload_templates() {
    print_status "Creating S3 bucket for CloudFormation templates..." >&2
    
    local bucket_name="${PROJECT_NAME}-cf-templates-${ENVIRONMENT}-$(aws sts get-caller-identity --query Account --output text)"
    
    # Create bucket if it doesn't exist
    if ! aws s3 ls "s3://$bucket_name" &> /dev/null; then
        aws s3 mb "s3://$bucket_name" --region $AWS_REGION >&2
        print_success "Created S3 bucket: $bucket_name" >&2
    else
        print_status "S3 bucket already exists: $bucket_name" >&2
    fi

    # Upload templates
    print_status "Uploading CloudFormation templates to S3..." >&2
    aws s3 sync infrastructure/cloudformation/ "s3://$bucket_name/templates/" --exclude "master-template.yaml" >&2
    
    print_success "Templates uploaded to S3" >&2
    echo "$bucket_name"
}

# Function to deploy the infrastructure
deploy_infrastructure() {
    local template_bucket=$1
    
    print_status "Deploying infrastructure stack: $STACK_NAME"
    
    # Update master template with S3 URLs
    local master_template="infrastructure/cloudformation/master-template-updated.yaml"
    sed -e "s|./s3-buckets.yaml|https://$template_bucket.s3.$AWS_REGION.amazonaws.com/templates/s3-buckets.yaml|g" \
        -e "s|./dynamodb-tables.yaml|https://$template_bucket.s3.$AWS_REGION.amazonaws.com/templates/dynamodb-tables.yaml|g" \
        -e "s|./api-gateway.yaml|https://$template_bucket.s3.$AWS_REGION.amazonaws.com/templates/api-gateway.yaml|g" \
        -e "s|./cloudfront.yaml|https://$template_bucket.s3.$AWS_REGION.amazonaws.com/templates/cloudfront.yaml|g" \
        infrastructure/cloudformation/master-template.yaml > $master_template

    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION &> /dev/null; then
        print_status "Stack exists. Updating..."
        aws cloudformation update-stack \
            --stack-name $STACK_NAME \
            --template-body file://$master_template \
            --parameters ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME \
                        ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
            --capabilities CAPABILITY_IAM \
            --region $AWS_REGION

        print_status "Waiting for stack update to complete..."
        aws cloudformation wait stack-update-complete --stack-name $STACK_NAME --region $AWS_REGION
    else
        print_status "Creating new stack..."
        aws cloudformation create-stack \
            --stack-name $STACK_NAME \
            --template-body file://$master_template \
            --parameters ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME \
                        ParameterKey=Environment,ParameterValue=$ENVIRONMENT \
            --capabilities CAPABILITY_IAM \
            --region $AWS_REGION \
            --tags Key=Project,Value=$PROJECT_NAME \
                   Key=Environment,Value=$ENVIRONMENT \
                   Key=ManagedBy,Value=CloudFormation

        print_status "Waiting for stack creation to complete..."
        aws cloudformation wait stack-create-complete --stack-name $STACK_NAME --region $AWS_REGION
    fi

    # Clean up temporary file
    rm -f $master_template

    print_success "Infrastructure deployment completed!"
}

# Function to display stack outputs
display_outputs() {
    print_status "Stack outputs:"
    aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $AWS_REGION \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
}

# Function to create environment configuration file
create_env_config() {
    print_status "Creating environment configuration file..."
    
    local config_file="infrastructure/config/${ENVIRONMENT}.json"
    mkdir -p infrastructure/config

    # Get stack outputs
    local outputs=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $AWS_REGION \
        --query 'Stacks[0].Outputs' \
        --output json)

    # Create configuration JSON
    cat > $config_file << EOF
{
  "environment": "$ENVIRONMENT",
  "region": "$AWS_REGION",
  "projectName": "$PROJECT_NAME",
  "stackName": "$STACK_NAME",
  "outputs": $outputs
}
EOF

    print_success "Environment configuration saved to: $config_file"
}

# Main execution
main() {
    print_status "Starting deployment for environment: $ENVIRONMENT in region: $AWS_REGION"
    
    # Pre-deployment checks
    check_aws_cli
    validate_templates
    
    # Upload templates and deploy
    template_bucket=$(upload_templates)
    deploy_infrastructure $template_bucket
    
    # Post-deployment tasks
    display_outputs
    create_env_config
    
    print_success "Deployment completed successfully!"
    print_status "Next steps:"
    echo "  1. Deploy Lambda functions"
    echo "  2. Update API Gateway with Lambda ARNs"
    echo "  3. Deploy React frontend to S3"
    echo "  4. Test the complete system"
}

# Script usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [environment] [region]"
        echo "  environment: dev, staging, or prod (default: dev)"
        echo "  region: AWS region (default: us-east-1)"
        echo ""
        echo "Example: $0 dev us-west-2"
        exit 0
    fi
    
    main "$@"
fi