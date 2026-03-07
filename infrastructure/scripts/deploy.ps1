# AI Data Analyst Platform Infrastructure Deployment Script (PowerShell)
# This script deploys the complete AWS infrastructure using CloudFormation

param(
    [string]$Environment = "dev",
    [string]$Region = "us-east-1",
    [switch]$Help
)

# Configuration
$ProjectName = "ai-data-analyst-platform"
$StackName = "$ProjectName-$Environment"

# Function to write colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to show help
function Show-Help {
    Write-Host "AI Data Analyst Platform Infrastructure Deployment"
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [-Environment <env>] [-Region <region>] [-Help]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Environment    Environment name (dev, staging, prod). Default: dev"
    Write-Host "  -Region         AWS region. Default: us-east-1"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1"
    Write-Host "  .\deploy.ps1 -Environment prod -Region us-west-2"
    exit 0
}

# Function to check if AWS CLI is installed and configured
function Test-AwsCli {
    try {
        $null = Get-Command aws -ErrorAction Stop
        $null = aws sts get-caller-identity 2>$null
        Write-Success "AWS CLI is configured"
        return $true
    }
    catch {
        Write-Error "AWS CLI is not installed or configured. Please install and configure it first."
        return $false
    }
}

# Function to validate CloudFormation templates
function Test-Templates {
    Write-Status "Validating CloudFormation templates..."
    
    $templates = @(
        "s3-buckets.yaml",
        "dynamodb-tables.yaml", 
        "api-gateway.yaml",
        "cloudfront.yaml",
        "master-template.yaml"
    )

    foreach ($template in $templates) {
        $templatePath = "infrastructure/cloudformation/$template"
        try {
            $null = aws cloudformation validate-template --template-body "file://$templatePath" 2>$null
            Write-Success "Template $template is valid"
        }
        catch {
            Write-Error "Template $template is invalid"
            return $false
        }
    }
    return $true
}

# Function to upload templates to S3
function Publish-Templates {
    Write-Status "Creating S3 bucket for CloudFormation templates..."
    
    $accountId = (aws sts get-caller-identity --query Account --output text)
    $bucketName = "$ProjectName-cf-templates-$Environment-$accountId"
    
    # Create bucket if it doesn't exist
    try {
        $null = aws s3 ls "s3://$bucketName" 2>$null
        Write-Status "S3 bucket already exists: $bucketName"
    }
    catch {
        aws s3 mb "s3://$bucketName" --region $Region
        Write-Success "Created S3 bucket: $bucketName"
    }

    # Upload templates
    Write-Status "Uploading CloudFormation templates to S3..."
    aws s3 sync infrastructure/cloudformation/ "s3://$bucketName/templates/" --exclude "master-template.yaml"
    
    Write-Success "Templates uploaded to S3"
    return $bucketName
}

# Function to deploy the infrastructure
function Deploy-Infrastructure {
    param([string]$TemplateBucket)
    
    Write-Status "Deploying infrastructure stack: $StackName"
    
    # Update master template with S3 URLs
    $masterTemplate = "infrastructure/cloudformation/master-template-updated.yaml"
    $content = Get-Content "infrastructure/cloudformation/master-template.yaml" -Raw
    $content = $content -replace "./s3-buckets.yaml", "https://$TemplateBucket.s3.$Region.amazonaws.com/templates/s3-buckets.yaml"
    $content = $content -replace "./dynamodb-tables.yaml", "https://$TemplateBucket.s3.$Region.amazonaws.com/templates/dynamodb-tables.yaml"
    $content = $content -replace "./api-gateway.yaml", "https://$TemplateBucket.s3.$Region.amazonaws.com/templates/api-gateway.yaml"
    $content = $content -replace "./cloudfront.yaml", "https://$TemplateBucket.s3.$Region.amazonaws.com/templates/cloudfront.yaml"
    $content | Set-Content $masterTemplate

    # Check if stack exists
    try {
        $null = aws cloudformation describe-stacks --stack-name $StackName --region $Region 2>$null
        Write-Status "Stack exists. Updating..."
        aws cloudformation update-stack `
            --stack-name $StackName `
            --template-body "file://$masterTemplate" `
            --parameters "ParameterKey=ProjectName,ParameterValue=$ProjectName" "ParameterKey=Environment,ParameterValue=$Environment" `
            --capabilities CAPABILITY_IAM `
            --region $Region

        Write-Status "Waiting for stack update to complete..."
        aws cloudformation wait stack-update-complete --stack-name $StackName --region $Region
    }
    catch {
        Write-Status "Creating new stack..."
        aws cloudformation create-stack `
            --stack-name $StackName `
            --template-body "file://$masterTemplate" `
            --parameters "ParameterKey=ProjectName,ParameterValue=$ProjectName" "ParameterKey=Environment,ParameterValue=$Environment" `
            --capabilities CAPABILITY_IAM `
            --region $Region `
            --tags "Key=Project,Value=$ProjectName" "Key=Environment,Value=$Environment" "Key=ManagedBy,Value=CloudFormation"

        Write-Status "Waiting for stack creation to complete..."
        aws cloudformation wait stack-create-complete --stack-name $StackName --region $Region
    }

    # Clean up temporary file
    if (Test-Path $masterTemplate) {
        Remove-Item $masterTemplate
    }

    Write-Success "Infrastructure deployment completed!"
}

# Function to display stack outputs
function Show-Outputs {
    Write-Status "Stack outputs:"
    aws cloudformation describe-stacks `
        --stack-name $StackName `
        --region $Region `
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' `
        --output table
}

# Function to create environment configuration file
function New-EnvConfig {
    Write-Status "Creating environment configuration file..."
    
    $configDir = "infrastructure/config"
    $configFile = "$configDir/$Environment.json"
    
    if (!(Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    # Get stack outputs
    $outputs = aws cloudformation describe-stacks `
        --stack-name $StackName `
        --region $Region `
        --query 'Stacks[0].Outputs' `
        --output json | ConvertFrom-Json

    # Create configuration object
    $config = @{
        environment = $Environment
        region = $Region
        projectName = $ProjectName
        stackName = $StackName
        outputs = $outputs
    }

    # Save configuration
    $config | ConvertTo-Json -Depth 10 | Set-Content $configFile
    Write-Success "Environment configuration saved to: $configFile"
}

# Main execution
function Main {
    if ($Help) {
        Show-Help
    }

    Write-Status "Starting deployment for environment: $Environment in region: $Region"
    
    # Pre-deployment checks
    if (!(Test-AwsCli)) {
        exit 1
    }
    
    if (!(Test-Templates)) {
        exit 1
    }
    
    # Upload templates and deploy
    $templateBucket = Publish-Templates
    Deploy-Infrastructure -TemplateBucket $templateBucket
    
    # Post-deployment tasks
    Show-Outputs
    New-EnvConfig
    
    Write-Success "Deployment completed successfully!"
    Write-Status "Next steps:"
    Write-Host "  1. Deploy Lambda functions"
    Write-Host "  2. Update API Gateway with Lambda ARNs"
    Write-Host "  3. Deploy React frontend to S3"
    Write-Host "  4. Test the complete system"
}

# Execute main function
Main