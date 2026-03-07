# AI Data Analyst Platform - Deployment Guide

This guide provides step-by-step instructions for deploying the complete AI Data Analyst Platform infrastructure on AWS.

## Overview

The AI Data Analyst Platform is a serverless web application that enables users to upload datasets, perform preprocessing operations, train machine learning models, and receive AI-powered analysis suggestions. The infrastructure is built using AWS CloudFormation and includes:

- **Frontend**: React application served via CloudFront and S3
- **Backend**: Lambda functions for data processing and ML operations
- **Storage**: S3 for data and DynamoDB for metadata
- **API**: API Gateway with REST endpoints
- **AI**: Amazon Bedrock for intelligent recommendations

## Prerequisites

### 1. AWS Account Setup

- Active AWS account with appropriate permissions
- AWS CLI installed and configured
- Sufficient service limits for the resources being created

### 2. Required AWS Permissions

Your AWS user/role needs permissions for:
- CloudFormation (full access)
- S3 (full access)
- DynamoDB (full access)
- API Gateway (full access)
- CloudFront (full access)
- Lambda (full access)
- IAM (for creating service roles)
- Bedrock (for AI functionality)

### 3. Local Development Environment

- **Operating System**: Linux, macOS, or Windows with WSL/PowerShell
- **AWS CLI**: Version 2.x recommended
- **Git**: For cloning the repository
- **Node.js**: Version 18+ (for frontend development)
- **Python**: Version 3.11+ (for Lambda functions)

## Phase 1: Infrastructure Foundation

### Step 1: Clone and Prepare

```bash
# Clone the repository
git clone <repository-url>
cd ai-data-analyst-platform

# Verify infrastructure files
ls -la infrastructure/
```

### Step 2: Validate Templates

```bash
# Linux/macOS
chmod +x infrastructure/scripts/*.sh
./infrastructure/scripts/validate.sh

# Windows PowerShell
.\infrastructure\scripts\validate.ps1
```

### Step 3: Deploy Infrastructure

```bash
# Deploy to development environment
./infrastructure/scripts/deploy.sh dev us-east-1

# Or using PowerShell on Windows
.\infrastructure\scripts\deploy.ps1 -Environment dev -Region us-east-1
```

**Expected Output:**
```
[INFO] Starting deployment for environment: dev in region: us-east-1
[SUCCESS] AWS CLI is configured
[SUCCESS] Template s3-buckets.yaml is valid
[SUCCESS] Template dynamodb-tables.yaml is valid
[SUCCESS] Template api-gateway.yaml is valid
[SUCCESS] Template cloudfront.yaml is valid
[SUCCESS] Template master-template.yaml is valid
[INFO] Creating S3 bucket for CloudFormation templates...
[SUCCESS] Created S3 bucket: ai-data-analyst-platform-cf-templates-dev-123456789012
[INFO] Uploading CloudFormation templates to S3...
[SUCCESS] Templates uploaded to S3
[INFO] Deploying infrastructure stack: ai-data-analyst-platform-dev
[INFO] Creating new stack...
[INFO] Waiting for stack creation to complete...
[SUCCESS] Infrastructure deployment completed!
```

### Step 4: Verify Deployment

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name ai-data-analyst-platform-dev

# List created resources
aws cloudformation list-stack-resources --stack-name ai-data-analyst-platform-dev
```

## Phase 2: Lambda Functions (Next Task)

After completing the infrastructure foundation, the next steps will be:

1. **Create Lambda Functions**: Implement the backend processing functions
2. **Deploy Lambda Code**: Package and deploy Python functions
3. **Update API Gateway**: Connect Lambda functions to API endpoints
4. **Configure Bedrock**: Set up AI integration and Guardrails

## Phase 3: Frontend Application (Future Task)

1. **Build React App**: Create the frontend application
2. **Deploy to S3**: Upload built files to the static website bucket
3. **Configure CloudFront**: Update distribution settings
4. **Test Integration**: Verify end-to-end functionality

## Configuration Files

After deployment, configuration files are created in `infrastructure/config/`:

```json
{
  "environment": "dev",
  "region": "us-east-1",
  "projectName": "ai-data-analyst-platform",
  "stackName": "ai-data-analyst-platform-dev",
  "outputs": [
    {
      "OutputKey": "StaticWebsiteBucketName",
      "OutputValue": "ai-data-analyst-platform-frontend-dev-123456789012"
    },
    {
      "OutputKey": "DataStorageBucketName", 
      "OutputValue": "ai-data-analyst-platform-data-dev-123456789012"
    },
    {
      "OutputKey": "APIGatewayURL",
      "OutputValue": "https://abc123.execute-api.us-east-1.amazonaws.com/dev"
    },
    {
      "OutputKey": "CloudFrontURL",
      "OutputValue": "https://d1234567890123.cloudfront.net"
    }
  ]
}
```

## Resource Overview

### Created AWS Resources

**S3 Buckets:**
- `ai-data-analyst-platform-frontend-dev-*`: Static website hosting
- `ai-data-analyst-platform-data-dev-*`: Data storage
- `ai-data-analyst-platform-cloudfront-logs-dev-*`: CloudFront access logs
- `ai-data-analyst-platform-cf-templates-dev-*`: CloudFormation templates

**DynamoDB Tables:**
- `ai-data-analyst-platform-sessions-dev`: Session management
- `ai-data-analyst-platform-operations-dev`: Operations logging
- `ai-data-analyst-platform-ai-decisions-dev`: AI recommendations
- `ai-data-analyst-platform-cache-dev`: Response caching

**API Gateway:**
- REST API with endpoints for upload, processing, training, and recommendations
- CORS configuration and rate limiting
- Usage plans and API keys

**CloudFront:**
- Global CDN distribution
- SSL/TLS termination
- Caching optimization
- Custom error pages for SPA routing

## Security Configuration

### Encryption
- **S3**: Server-side encryption (AES-256)
- **DynamoDB**: Encryption at rest
- **API Gateway**: HTTPS enforced
- **CloudFront**: SSL/TLS certificates

### Access Control
- **S3**: Origin Access Control (OAC) for CloudFront
- **DynamoDB**: IAM-based access
- **API Gateway**: Rate limiting (100 req/min)
- **Lambda**: Minimal privilege roles (to be configured)

## Monitoring Setup

### CloudWatch Integration
- All services configured for CloudWatch logging
- Metrics collection enabled
- Performance monitoring ready

### Recommended Alarms
```bash
# Create basic alarms (example)
aws cloudwatch put-metric-alarm \
  --alarm-name "API-Gateway-4XX-Errors" \
  --alarm-description "API Gateway 4XX error rate" \
  --metric-name 4XXError \
  --namespace AWS/ApiGateway \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold
```

## Cost Optimization

### Current Configuration
- **Pay-per-use**: Lambda, API Gateway, DynamoDB on-demand
- **Lifecycle policies**: S3 automatic cleanup after 30 days
- **Caching**: CloudFront and DynamoDB cache table
- **Compression**: Enabled for all static assets

### Estimated Monthly Costs (1000 users)
- **Lambda**: $50-100
- **S3**: $20-40  
- **DynamoDB**: $30-60
- **API Gateway**: $30-50
- **CloudFront**: $10-20
- **Bedrock**: $100-200
- **Total**: $240-470/month

## Troubleshooting

### Common Issues

**1. Stack Creation Fails**
```bash
# Check stack events for detailed errors
aws cloudformation describe-stack-events --stack-name ai-data-analyst-platform-dev
```

**2. Permission Errors**
- Verify AWS CLI configuration: `aws sts get-caller-identity`
- Check IAM permissions for CloudFormation and related services
- Ensure service limits are not exceeded

**3. Resource Name Conflicts**
- S3 bucket names must be globally unique
- If deployment fails due to existing names, modify the ProjectName parameter

**4. Template Validation Errors**
```bash
# Validate individual templates
aws cloudformation validate-template --template-body file://infrastructure/cloudformation/s3-buckets.yaml
```

### Debugging Commands

```bash
# Check resource status
aws cloudformation describe-stack-resources --stack-name ai-data-analyst-platform-dev

# View CloudFormation events
aws cloudformation describe-stack-events --stack-name ai-data-analyst-platform-dev --max-items 20

# Test S3 bucket access
aws s3 ls s3://ai-data-analyst-platform-frontend-dev-123456789012

# Test API Gateway
curl -X GET https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/
```

## Cleanup

To remove all resources:

```bash
# Linux/macOS
./infrastructure/scripts/destroy.sh dev us-east-1

# Windows PowerShell  
.\infrastructure\scripts\destroy.ps1 -Environment dev -Region us-east-1
```

**Warning**: This permanently deletes all data and resources!

## Next Steps

1. **Complete Task 1**: Verify infrastructure deployment is successful
2. **Proceed to Task 2**: Implement core data processing infrastructure (Lambda functions)
3. **Continue with Tasks 3-15**: Follow the implementation plan in tasks.md
4. **Test Integration**: Verify each component works correctly
5. **Deploy to Production**: Repeat process for production environment

## Support

For deployment issues:
1. Check CloudFormation stack events for detailed error messages
2. Review AWS service quotas and limits
3. Verify IAM permissions are sufficient
4. Consult AWS documentation for service-specific guidance
5. Use AWS Support if you have a support plan

## Success Criteria

Infrastructure deployment is successful when:
- ✅ All CloudFormation stacks deploy without errors
- ✅ S3 buckets are created and accessible
- ✅ DynamoDB tables are created with correct schema
- ✅ API Gateway endpoints return 404 (expected without Lambda functions)
- ✅ CloudFront distribution is active and serving content
- ✅ Configuration files are generated correctly
- ✅ All resources are tagged appropriately

The infrastructure foundation is now ready for Lambda function deployment and frontend integration.