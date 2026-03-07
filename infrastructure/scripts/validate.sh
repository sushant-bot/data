#!/bin/bash

# AI Data Analyst Platform Infrastructure Validation Script
# This script validates all CloudFormation templates

set -e

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to validate a single template
validate_template() {
    local template_path=$1
    local template_name=$(basename "$template_path")
    
    print_status "Validating $template_name..."
    
    if aws cloudformation validate-template --template-body "file://$template_path" > /dev/null 2>&1; then
        print_success "$template_name is valid"
        return 0
    else
        print_error "$template_name is invalid"
        # Show the actual error
        aws cloudformation validate-template --template-body "file://$template_path" 2>&1 || true
        return 1
    fi
}

# Main validation function
main() {
    print_status "Starting CloudFormation template validation..."
    
    # Check if AWS CLI is available
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi

    # Check if AWS is configured
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi

    local templates=(
        "infrastructure/cloudformation/s3-buckets.yaml"
        "infrastructure/cloudformation/dynamodb-tables.yaml"
        "infrastructure/cloudformation/api-gateway.yaml"
        "infrastructure/cloudformation/cloudfront.yaml"
        "infrastructure/cloudformation/master-template.yaml"
    )

    local failed=0
    local total=${#templates[@]}

    for template in "${templates[@]}"; do
        if [[ -f "$template" ]]; then
            if ! validate_template "$template"; then
                ((failed++))
            fi
        else
            print_error "Template not found: $template"
            ((failed++))
        fi
    done

    echo ""
    if [[ $failed -eq 0 ]]; then
        print_success "All $total templates are valid!"
        exit 0
    else
        print_error "$failed out of $total templates failed validation"
        exit 1
    fi
}

# Script usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0"
        echo "Validates all CloudFormation templates in the infrastructure directory"
        exit 0
    fi
    
    main "$@"
fi