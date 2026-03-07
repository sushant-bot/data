#!/bin/bash

# AI Data Analyst Platform Infrastructure Destruction Script
# This script safely destroys the AWS infrastructure

set -e

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

# Function to confirm destruction
confirm_destruction() {
    print_warning "This will permanently delete all resources in the $ENVIRONMENT environment!"
    print_warning "This includes:"
    echo "  - S3 buckets and all data"
    echo "  - DynamoDB tables and all data"
    echo "  - API Gateway"
    echo "  - CloudFront distribution"
    echo "  - All associated logs and configurations"
    echo ""
    
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirmation
    
    if [[ "$confirmation" != "yes" ]]; then
        print_status "Destruction cancelled."
        exit 0
    fi
}

# Function to empty S3 buckets before deletion
empty_s3_buckets() {
    print_status "Emptying S3 buckets before deletion..."
    
    # Get bucket names from stack outputs
    local static_bucket=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $AWS_REGION \
        --query 'Stacks[0].Outputs[?OutputKey==`StaticWebsiteBucketName`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    local data_bucket=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $AWS_REGION \
        --query 'Stacks[0].Outputs[?OutputKey==`DataStorageBucketName`].OutputValue' \
        --output text 2>/dev/null || echo "")

    # Empty buckets if they exist
    if [[ -n "$static_bucket" ]] && aws s3 ls "s3://$static_bucket" &> /dev/null; then
        print_status "Emptying static website bucket: $static_bucket"
        aws s3 rm "s3://$static_bucket" --recursive
        print_success "Static website bucket emptied"
    fi

    if [[ -n "$data_bucket" ]] && aws s3 ls "s3://$data_bucket" &> /dev/null; then
        print_status "Emptying data storage bucket: $data_bucket"
        aws s3 rm "s3://$data_bucket" --recursive
        print_success "Data storage bucket emptied"
    fi

    # Empty CloudFront logs bucket
    local cf_logs_bucket="${PROJECT_NAME}-cloudfront-logs-${ENVIRONMENT}-$(aws sts get-caller-identity --query Account --output text)"
    if aws s3 ls "s3://$cf_logs_bucket" &> /dev/null; then
        print_status "Emptying CloudFront logs bucket: $cf_logs_bucket"
        aws s3 rm "s3://$cf_logs_bucket" --recursive
        print_success "CloudFront logs bucket emptied"
    fi

    # Empty templates bucket
    local templates_bucket="${PROJECT_NAME}-cf-templates-${ENVIRONMENT}-$(aws sts get-caller-identity --query Account --output text)"
    if aws s3 ls "s3://$templates_bucket" &> /dev/null; then
        print_status "Emptying templates bucket: $templates_bucket"
        aws s3 rm "s3://$templates_bucket" --recursive
        aws s3 rb "s3://$templates_bucket"
        print_success "Templates bucket deleted"
    fi
}

# Function to delete the CloudFormation stack
delete_stack() {
    print_status "Deleting CloudFormation stack: $STACK_NAME"
    
    # Check if stack exists
    if ! aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION &> /dev/null; then
        print_warning "Stack $STACK_NAME does not exist"
        return 0
    fi

    # Delete the stack
    aws cloudformation delete-stack --stack-name $STACK_NAME --region $AWS_REGION
    
    print_status "Waiting for stack deletion to complete..."
    aws cloudformation wait stack-delete-complete --stack-name $STACK_NAME --region $AWS_REGION
    
    print_success "Stack deleted successfully"
}

# Function to clean up local files
cleanup_local_files() {
    print_status "Cleaning up local configuration files..."
    
    local config_file="infrastructure/config/${ENVIRONMENT}.json"
    if [[ -f "$config_file" ]]; then
        rm "$config_file"
        print_success "Removed configuration file: $config_file"
    fi
    
    # Remove temporary files
    rm -f infrastructure/cloudformation/master-template-updated.yaml
}

# Function to verify deletion
verify_deletion() {
    print_status "Verifying resource deletion..."
    
    # Check if stack still exists
    if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION &> /dev/null; then
        print_error "Stack still exists. Deletion may have failed."
        return 1
    fi
    
    print_success "All resources have been successfully deleted"
}

# Main execution
main() {
    print_status "Starting destruction for environment: $ENVIRONMENT in region: $AWS_REGION"
    
    # Pre-destruction checks
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured"
        exit 1
    fi

    # Confirm destruction
    confirm_destruction
    
    # Execute destruction steps
    empty_s3_buckets
    delete_stack
    cleanup_local_files
    verify_deletion
    
    print_success "Infrastructure destruction completed successfully!"
}

# Script usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [environment] [region]"
        echo "  environment: dev, staging, or prod (default: dev)"
        echo "  region: AWS region (default: us-east-1)"
        echo ""
        echo "Example: $0 dev us-west-2"
        echo ""
        echo "WARNING: This will permanently delete all resources!"
        exit 0
    fi
    
    main "$@"
fi