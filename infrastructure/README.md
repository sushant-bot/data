# AI Data Analyst Platform - AWS Infrastructure

This directory contains the complete AWS infrastructure setup for the AI Data Analyst Platform using CloudFormation templates and deployment scripts.

## Architecture Overview

The platform uses a serverless architecture with the following AWS services:

- **Amazon S3**: Static website hosting and data storage
- **Amazon DynamoDB**: Session management, operations logging, AI decisions, and caching
- **Amazon API Gateway**: REST API endpoints with CORS and rate limiting
- **Amazon CloudFront**: Global content delivery network
- **AWS Lambda**: Backend processing functions (to be deployed separately)
- **Amazon Bedrock**: AI-powered analysis recommendations
- **Amazon CloudWatch**: Monitoring and logging

## Directory Structure

```
infrastructure/
├── cloudformation/
│   ├── s3-buckets.yaml          # S3 buckets for frontend and data storage
│   ├── dynamodb-tables.yaml     # DynamoDB tables for data persistence
│   ├── api-gateway.yaml         # API Gateway with REST endpoints
│   ├── cloudfront.yaml          # CloudFront distribution
│   └── master-template.yaml     # Master template orchestrating all stacks
├── scripts/
│   ├── deploy.sh               # Deployment script
│   └── destroy.sh              # Destruction script
├── config/                     # Environment-specific configurations (generated)
└── README.md                   # This file
```

## Prerequisites

1. **AWS CLI**: Install and configure the AWS CLI
   ```bash
   aws configure
   ```

2. **AWS Permissions**: Ensure your AWS credentials have permissions for:
   - CloudFormation (full access)
   - S3 (full access)
   - DynamoDB (full access)
   - API Gateway (full access)
   - CloudFront (full access)
   - IAM (for service roles)

3. **Bash Shell**: The deployment scripts require bash (Linux/macOS/WSL)

## Quick Start

### Deploy Infrastructure

```bash
# Make scripts executable
chmod +x infrastructure/scripts/*.sh

# Deploy to development environment
./infrastructure/scripts/deploy.sh dev us-east-1

# Deploy to production environment
./infrastructure/scripts/deploy.sh prod us-east-1
```

### Destroy Infrastructure

```bash
# Destroy development environment
./infrastructure/scripts/destroy.sh dev us-east-1
```

## Detailed Component Documentation

### S3 Buckets

**Static Website Bucket**:
- Hosts the React frontend application
- Configured for static website hosting
- Public read access via CloudFront
- Versioning enabled for rollback capability

**Data Storage Bucket**:
- Stores uploaded datasets, processed data, and visualizations
- Private access (Lambda functions only)
- 30-day lifecycle policy with IA transition after 7 days
- Server-side encryption enabled

### DynamoDB Tables

**Sessions Table**:
- **Partition Key**: `session_id` (String)
- **Purpose**: Store session metadata and workflow state
- **Attributes**: timestamp, dataset_name, file_size, status, quality_score

**Operations Table**:
- **Partition Key**: `session_id` (String)
- **Sort Key**: `timestamp` (String)
- **Purpose**: Log all preprocessing and training operations
- **TTL**: Enabled for automatic cleanup

**AI Decisions Table**:
- **Partition Key**: `session_id` (String)
- **Sort Key**: `decision_type` (String)
- **Purpose**: Store AI recommendations and reasoning
- **TTL**: Enabled for automatic cleanup

**Cache Table**:
- **Partition Key**: `prompt_hash` (String)
- **Purpose**: Cache AI responses for performance optimization
- **TTL**: 24 hours for automatic cleanup

### API Gateway

**Endpoints**:
- `POST /upload` - Dataset upload
- `GET /preview/{sessionId}` - Dataset preview
- `POST /preprocess` - Data preprocessing operations
- `POST /train` - Model training
- `GET /recommendations/{sessionId}` - AI recommendations
- `GET /visualizations/{sessionId}` - Generated charts
- `GET /sessions/{sessionId}` - Session details

**Features**:
- CORS configuration for frontend integration
- Rate limiting: 100 requests/minute per IP
- Request validation and error handling
- Usage plans and API keys for monitoring

### CloudFront Distribution

**Configuration**:
- Global edge locations for low latency
- HTTPS redirect enforced
- Caching optimized for static assets
- Custom error pages for SPA routing
- Access logging enabled

**Cache Behaviors**:
- Static assets (`/static/*`): Long-term caching
- API calls (`/api/*`): No caching
- Default: Optimized caching with compression

## Environment Configuration

After deployment, configuration files are automatically generated:

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
      "OutputKey": "APIGatewayURL",
      "OutputValue": "https://abc123.execute-api.us-east-1.amazonaws.com/dev"
    }
  ]
}
```

## Security Features

### Encryption
- S3 buckets: Server-side encryption (AES-256)
- DynamoDB: Encryption at rest enabled
- API Gateway: HTTPS/TLS 1.2 enforced
- CloudFront: SSL/TLS certificates

### Access Control
- S3 buckets: Minimal public access, CloudFront OAC
- DynamoDB: IAM-based access control
- API Gateway: Rate limiting and usage plans
- Lambda: Minimal privilege IAM roles (to be configured)

### Monitoring
- CloudWatch integration for all services
- Access logging for S3 and CloudFront
- API Gateway request/response logging
- DynamoDB streams for audit trails

## Cost Optimization

### Pay-per-Use Services
- Lambda: No idle costs
- DynamoDB: On-demand billing
- API Gateway: Pay per request
- S3: Storage and request-based pricing

### Lifecycle Policies
- S3: Automatic transition to IA after 7 days, deletion after 30 days
- DynamoDB: TTL for automatic data cleanup
- CloudFront logs: 90-day retention with archival

### Caching Strategy
- CloudFront: Reduces origin requests
- DynamoDB Cache table: Reduces Bedrock API calls
- API Gateway: Response caching where appropriate

## Monitoring and Alerting

### CloudWatch Metrics
- API Gateway: Request count, latency, errors
- S3: Storage metrics, request metrics
- DynamoDB: Read/write capacity, throttling
- CloudFront: Cache hit ratio, origin latency

### Recommended Alarms
- API Gateway error rate > 5%
- DynamoDB throttling events
- S3 4xx/5xx error rate
- CloudFront origin errors

## Troubleshooting

### Common Issues

**Stack Creation Fails**:
- Check IAM permissions
- Verify AWS CLI configuration
- Ensure unique resource names

**S3 Bucket Policy Errors**:
- Bucket names must be globally unique
- Check region consistency
- Verify CloudFront OAC configuration

**API Gateway CORS Issues**:
- Verify OPTIONS methods are configured
- Check response headers configuration
- Ensure frontend domain is allowed

### Debugging Commands

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name ai-data-analyst-platform-dev

# View stack events
aws cloudformation describe-stack-events --stack-name ai-data-analyst-platform-dev

# Check S3 bucket contents
aws s3 ls s3://your-bucket-name --recursive

# Test API Gateway endpoint
curl -X GET https://your-api-id.execute-api.region.amazonaws.com/dev/sessions/test-session
```

## Next Steps

After infrastructure deployment:

1. **Deploy Lambda Functions**: Create and deploy the backend processing functions
2. **Update API Gateway**: Connect Lambda functions to API Gateway endpoints
3. **Deploy Frontend**: Build and deploy the React application to S3
4. **Configure Bedrock**: Set up Amazon Bedrock access and Guardrails
5. **Test Integration**: Verify end-to-end functionality

## Support

For issues or questions:
1. Check CloudFormation stack events for detailed error messages
2. Review CloudWatch logs for runtime issues
3. Verify IAM permissions and service quotas
4. Consult AWS documentation for service-specific guidance

## Cost Estimation

**Monthly costs for 1000 active users** (approximate):
- Lambda: $50-100
- S3: $20-40
- DynamoDB: $30-60
- API Gateway: $30-50
- CloudFront: $10-20
- Bedrock: $100-200
- **Total**: $240-470/month

Costs scale with usage and can be optimized through caching and lifecycle policies.