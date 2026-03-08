"""Lambda Functions Deployment Script for AI Data Analyst Platform"""
import subprocess
import json
import os
import sys
import zipfile
import shutil
import time

PROJECT_NAME = "ai-data-analyst-platform"
ENVIRONMENT = sys.argv[1] if len(sys.argv) > 1 else "dev"
AWS_REGION = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"

# Get project root (parent of infrastructure/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)

# Packages that are already in Lambda runtime or will be in layers
RUNTIME_PACKAGES = ['boto3', 'botocore', 's3transfer', 'urllib3', 'jmespath', 'dateutil', 'six']
# Heavy packages to put in layers
LAYER_PACKAGES = ['pandas', 'numpy', 'sklearn', 'scipy', 'scipy.libs', 'matplotlib', 'seaborn',
                  'mpl_toolkits', 'PIL', 'Pillow', 'fontTools', 'kiwisolver', 'contourpy',
                  'cycler', 'pyparsing', 'packaging', 'joblib', 'threadpoolctl']

FUNCTIONS = [
    {"dir": "upload", "name": f"{PROJECT_NAME}-upload-{ENVIRONMENT}", "memory": 1024, "timeout": 300},
    {"dir": "preview", "name": f"{PROJECT_NAME}-preview-{ENVIRONMENT}", "memory": 1024, "timeout": 300},
    {"dir": "processing", "name": f"{PROJECT_NAME}-processing-{ENVIRONMENT}", "memory": 2048, "timeout": 600},
    {"dir": "ml_training", "name": f"{PROJECT_NAME}-ml-training-{ENVIRONMENT}", "memory": 2048, "timeout": 900},
    {"dir": "visualization", "name": f"{PROJECT_NAME}-visualization-{ENVIRONMENT}", "memory": 1024, "timeout": 300},
    {"dir": "ai_assistant", "name": f"{PROJECT_NAME}-ai-assistant-{ENVIRONMENT}", "memory": 1024, "timeout": 300},
]


def run(cmd, capture=False):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if capture:
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            return None
        return result.stdout.strip()
    if result.returncode != 0:
        print(f"  Command failed with code {result.returncode}")
        sys.exit(1)


def fpath(p):
    """Convert Windows backslash path to forward slashes for AWS CLI file:// URIs."""
    return p.replace("\\", "/")


def get_account_id():
    return run("aws sts get-caller-identity --query Account --output text", capture=True)


def create_layer(layer_name, packages, account_id, exclude_pkgs=None):
    """Create a Lambda layer with specified packages."""
    print(f"\n--- Creating Layer: {layer_name} ---")

    build_dir = os.path.join(PROJECT_ROOT, f"build-layer-{layer_name}")
    # Lambda layers must have python/ directory
    python_dir = os.path.join(build_dir, "python")
    zip_path = os.path.join(PROJECT_ROOT, f"{layer_name}.zip")

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(python_dir)

    # Install packages
    pkg_str = " ".join(packages)
    print(f"  Installing: {pkg_str}")
    result = subprocess.run(
        f'pip install --no-user {pkg_str} -t "{python_dir}" '
        f'--platform manylinux2014_x86_64 --implementation cp --python-version 3.11 '
        f'--only-binary=:all: --upgrade --quiet',
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        subprocess.run(
            f'pip install --no-user {pkg_str} -t "{python_dir}" --quiet',
            shell=True, capture_output=True, text=True
        )

    # Remove boto3/botocore from layer + any excluded packages
    remove_pkgs = list(RUNTIME_PACKAGES)
    if exclude_pkgs:
        remove_pkgs.extend(exclude_pkgs)
    for pkg in remove_pkgs:
        for d in os.listdir(python_dir):
            if d.startswith(pkg) or d == pkg:
                p = os.path.join(python_dir, d)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)

    # Aggressive cleanup
    for root, dirs, files in os.walk(python_dir, topdown=True):
        for d in list(dirs):
            dl = d.lower()
            if dl in ('__pycache__', 'tests', 'test', 'testing', 'docs', 'doc',
                       'examples', 'example', 'benchmarks', '.dist-info',
                       'benchmark', '_benchmarks', 'conftest'):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
        # Remove unnecessary files
        for f in files:
            fl = f.lower()
            fp = os.path.join(root, f)
            if fl.endswith(('.pyc', '.pyo', '.pyi', '.c', '.h', '.hpp', '.rst',
                            '.md', '.txt', '.html', '.css')):
                os.remove(fp)

    # Strip .so files to reduce size (if strip is available)
    try:
        for root, dirs, files in os.walk(python_dir):
            for f in files:
                if f.endswith('.so'):
                    fp = os.path.join(root, f)
                    subprocess.run(f'strip --strip-unneeded "{fp}"',
                                   shell=True, capture_output=True)
    except Exception:
        pass

    # Check size
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(build_dir)
        for f in files
    )
    print(f"  Unzipped layer size: {total_size / (1024*1024):.1f} MB")

    if total_size > 250 * 1024 * 1024:
        print(f"  WARNING: Layer exceeds 250MB limit, further trimming needed")

    # Create ZIP
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(build_dir):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for f in files:
                full_path = os.path.join(root, f)
                arc_name = os.path.relpath(full_path, build_dir)
                zf.write(full_path, arc_name)

    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  ZIP size: {zip_mb:.1f} MB")

    if zip_mb < 0.001:
        print(f"  WARNING: Layer zip is empty (packages failed to install). Skipping layer publish.")
        shutil.rmtree(build_dir)
        os.remove(zip_path)
        return None
    s3_bucket = f"{PROJECT_NAME}-data-{ENVIRONMENT}-{account_id}"
    s3_key = f"lambda-layers/{layer_name}.zip"
    print(f"  Uploading layer to S3...")
    run(f'aws s3 cp "{zip_path}" "s3://{s3_bucket}/{s3_key}" --region {AWS_REGION}')

    layer_arn = run(
        f'aws lambda publish-layer-version --layer-name {layer_name} '
        f'--content S3Bucket={s3_bucket},S3Key={s3_key} '
        f'--compatible-runtimes python3.11 '
        f'--region {AWS_REGION} --query LayerVersionArn --output text',
        capture=True
    )

    print(f"  Layer ARN: {layer_arn}")

    # Cleanup
    shutil.rmtree(build_dir)
    os.remove(zip_path)

    return layer_arn


def create_lambda_role(account_id):
    role_name = f"{PROJECT_NAME}-lambda-role-{ENVIRONMENT}"

    # Check if role exists
    check = run(f"aws iam get-role --role-name {role_name} --query Role.Arn --output text", capture=True)
    if check and "arn:aws:iam" in check:
        print(f"  IAM role already exists: {check}")
        return check

    print("  Creating IAM role...")
    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    })

    # Write trust policy to temp file to avoid shell quoting issues on Windows
    trust_policy_file = os.path.join(PROJECT_ROOT, "tmp-trust-policy.json")
    with open(trust_policy_file, "w") as f:
        f.write(trust_policy)
    run(f'aws iam create-role --role-name {role_name} '
        f'--assume-role-policy-document "file://{fpath(trust_policy_file)}" '
        f'--output text --query Role.Arn')
    os.remove(trust_policy_file)

    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

    # Attach basic execution
    run(f'aws iam attach-role-policy --role-name {role_name} '
        f'--policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')

    # Custom policy
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
                "Resource": [
                    f"arn:aws:s3:::{PROJECT_NAME}-data-{ENVIRONMENT}-{account_id}",
                    f"arn:aws:s3:::{PROJECT_NAME}-data-{ENVIRONMENT}-{account_id}/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query", "dynamodb:Scan"],
                "Resource": [
                    f"arn:aws:dynamodb:{AWS_REGION}:{account_id}:table/{PROJECT_NAME}-*-{ENVIRONMENT}",
                    f"arn:aws:dynamodb:{AWS_REGION}:{account_id}:table/{PROJECT_NAME}-*-{ENVIRONMENT}/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel", "bedrock:ApplyGuardrail"],
                "Resource": "*"
            }
        ]
    })

    policy_name = f"{PROJECT_NAME}-lambda-policy-{ENVIRONMENT}"
    # Write policy to temp file to avoid shell escaping issues
    policy_file = os.path.join(PROJECT_ROOT, "tmp-policy.json")
    with open(policy_file, "w") as f:
        f.write(policy)

    run(f'aws iam put-role-policy --role-name {role_name} '
        f'--policy-name {policy_name} '
        f'--policy-document "file://{fpath(policy_file)}"')

    os.remove(policy_file)

    print("  Waiting 10s for IAM role propagation...")
    time.sleep(10)

    print(f"  Created role: {role_arn}")
    return role_arn


def package_function(func):
    """Package a Lambda function into a ZIP file."""
    dir_name = func["dir"]
    func_name = func["name"]

    build_dir = os.path.join(PROJECT_ROOT, f"build-{dir_name}")
    zip_path = os.path.join(PROJECT_ROOT, f"{func_name}.zip")

    # Clean up
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    # Copy lambda code
    lambda_dir = os.path.join(PROJECT_ROOT, "lambda", dir_name)
    for f in os.listdir(lambda_dir):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(lambda_dir, f), build_dir)

    # Copy shared utilities
    shared_dir = os.path.join(PROJECT_ROOT, "lambda", "shared")
    if os.path.exists(shared_dir):
        shared_dest = os.path.join(build_dir, "shared")
        os.makedirs(shared_dest, exist_ok=True)
        for f in os.listdir(shared_dir):
            if f.endswith(".py"):
                shutil.copy2(os.path.join(shared_dir, f), shared_dest)

    # Install deps for Linux target (layers handle heavy packages)
    req_file = os.path.join(lambda_dir, "requirements.txt")
    if os.path.exists(req_file):
        print(f"  Installing dependencies...")
        result = subprocess.run(
            f'pip install --no-user -r "{req_file}" -t "{build_dir}" '
            f'--platform manylinux2014_x86_64 --implementation cp --python-version 3.11 '
            f'--only-binary=:all: --upgrade --quiet',
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            subprocess.run(
                f'pip install --no-user -r "{req_file}" -t "{build_dir}" --quiet',
                shell=True, capture_output=True, text=True
            )

    # Remove packages that are in layers or Lambda runtime
    all_skip = RUNTIME_PACKAGES + LAYER_PACKAGES
    for item in os.listdir(build_dir):
        item_lower = item.lower().replace('-', '_')
        item_path = os.path.join(build_dir, item)
        for pkg in all_skip:
            if item_lower.startswith(pkg.lower()) or item_lower == pkg.lower():
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path):
                    os.remove(item_path)
                break

    # Create ZIP
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Trim unnecessary files
    skip_patterns = ['__pycache__', 'tests', 'test', '.dist-info', 'docs', 'doc',
                     'examples', 'example', 'benchmarks', 'benchmark']
    for root, dirs, files in os.walk(build_dir, topdown=True):
        for d in list(dirs):
            if any(p in d.lower() for p in skip_patterns):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(build_dir):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for f in files:
                full_path = os.path.join(root, f)
                arc_name = os.path.relpath(full_path, build_dir)
                zf.write(full_path, arc_name)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  Package size: {size_mb:.1f} MB")

    # Clean up build dir
    shutil.rmtree(build_dir)

    return zip_path


def deploy_function(func, role_arn, account_id, layer_arns=None):
    """Deploy a single Lambda function."""
    func_name = func["name"]
    dir_name = func["dir"]
    print(f"\n--- Deploying: {func_name} (from lambda/{dir_name}) ---")

    zip_path = package_function(func)

    # Environment variables
    env_vars = {
        "DATA_BUCKET": f"{PROJECT_NAME}-data-{ENVIRONMENT}-{account_id}",
        "BUCKET_NAME": f"{PROJECT_NAME}-data-{ENVIRONMENT}-{account_id}",
        "SESSIONS_TABLE": f"{PROJECT_NAME}-sessions-{ENVIRONMENT}",
        "OPERATIONS_TABLE": f"{PROJECT_NAME}-operations-{ENVIRONMENT}",
        "AI_DECISIONS_TABLE": f"{PROJECT_NAME}-ai-decisions-{ENVIRONMENT}",
        "CACHE_TABLE": f"{PROJECT_NAME}-cache-{ENVIRONMENT}",
    }
    if dir_name == "ai_assistant":
        env_vars["BEDROCK_MODEL_ID"] = "anthropic.claude-3-haiku-20240307-v1:0"

    env_json = json.dumps({"Variables": env_vars})
    env_file = os.path.join(PROJECT_ROOT, "tmp-env.json")
    with open(env_file, "w") as f:
        f.write(env_json)

    # Check if function exists
    exists = run(f'aws lambda get-function --function-name {func_name} --region {AWS_REGION}', capture=True)

    zip_size = os.path.getsize(zip_path)
    layers_arg = ""
    if layer_arns:
        valid_arns = [a for a in layer_arns if a]
        if valid_arns:
            layers_arg = f'--layers {" ".join(valid_arns)}'

    if exists and "FunctionArn" in exists:
        print(f"  Updating existing function...")
        run(f'aws lambda update-function-code --function-name {func_name} '
            f'--zip-file "fileb://{fpath(zip_path)}" --region {AWS_REGION} --output text --query FunctionArn')

        # Wait for code update
        run(f'aws lambda wait function-updated-v2 --function-name {func_name} --region {AWS_REGION}', capture=True)

        run(f'aws lambda update-function-configuration --function-name {func_name} '
            f'--memory-size {func["memory"]} --timeout {func["timeout"]} '
            f'--environment "file://{fpath(env_file)}" {layers_arg} --region {AWS_REGION} --output text --query FunctionArn')
    else:
        print(f"  Creating new function...")

        if zip_size > 50 * 1024 * 1024:
            # Upload to S3 first
            s3_bucket = f"{PROJECT_NAME}-data-{ENVIRONMENT}-{account_id}"
            s3_key = f"lambda-packages/{func_name}.zip"
            print(f"  Package > 50MB, uploading to S3 first...")
            run(f'aws s3 cp "{zip_path}" "s3://{s3_bucket}/{s3_key}" --region {AWS_REGION}')
            code_arg = f'S3Bucket={s3_bucket},S3Key={s3_key}'
            run(f'aws lambda create-function --function-name {func_name} '
                f'--runtime python3.11 --role "{role_arn}" '
                f'--handler lambda_function.lambda_handler '
                f'--memory-size {func["memory"]} --timeout {func["timeout"]} '
                f'--code {code_arg} '
                f'--environment "file://{fpath(env_file)}" {layers_arg} '
                f'--region {AWS_REGION} --output text --query FunctionArn')
        else:
            run(f'aws lambda create-function --function-name {func_name} '
                f'--runtime python3.11 --role "{role_arn}" '
                f'--handler lambda_function.lambda_handler '
                f'--memory-size {func["memory"]} --timeout {func["timeout"]} '
                f'--zip-file "fileb://{fpath(zip_path)}" '
                f'--environment "file://{env_file}" {layers_arg} '
                f'--region {AWS_REGION} --output text --query FunctionArn')

    # Wait for function to be active
    print(f"  Waiting for function to be active...")
    run(f'aws lambda wait function-active-v2 --function-name {func_name} --region {AWS_REGION}', capture=True)

    # Clean up
    os.remove(zip_path)
    if os.path.exists(env_file):
        os.remove(env_file)

    # Get ARN
    arn = run(f'aws lambda get-function --function-name {func_name} --region {AWS_REGION} '
              f'--query Configuration.FunctionArn --output text', capture=True)
    print(f"  Deployed: {arn}")
    return arn


def main():
    print("=" * 60)
    print("Deploying Lambda Functions")
    print(f"Project: {PROJECT_NAME}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Region: {AWS_REGION}")
    print("=" * 60)

    account_id = get_account_id()
    print(f"Account: {account_id}")

    print("\n--- Setting up IAM Role ---")
    role_arn = create_lambda_role(account_id)

    # Use AWS-provided pandas layer (AWSSDKPandas-Python311) instead of building our own
    # This avoids Windows cross-compilation issues and is the recommended approach
    print("\n--- Resolving Lambda Layers ---")
    aws_pandas_layer = run(
        f'aws lambda list-layers --region {AWS_REGION} --compatible-runtime python3.11 '
        f'--query "Layers[?LayerName==\'AWSSDKPandas-Python311\'].LatestMatchingVersion.LayerVersionArn | [0]" '
        f'--output text',
        capture=True
    )
    if aws_pandas_layer and aws_pandas_layer != 'None':
        print(f"  Found AWS pandas layer: {aws_pandas_layer}")
        data_layer_arn = aws_pandas_layer
    else:
        print("  AWS pandas layer not found, building custom data layer...")
        data_layer_arn = create_layer(
            f"{PROJECT_NAME}-data-layer-{ENVIRONMENT}",
            ["pandas==2.1.4", "numpy==1.24.3", "pytz"],
            account_id
        )

    ml_layer_arn = create_layer(
        f"{PROJECT_NAME}-ml-layer-{ENVIRONMENT}",
        ["scikit-learn==1.3.2", "joblib", "threadpoolctl"],
        account_id,
        exclude_pkgs=["numpy", "numpy.libs", "pandas", "pytz", "tzdata"]
    )

    # Map functions to their layers
    layer_map = {
        "upload": [data_layer_arn],
        "preview": [data_layer_arn],
        "processing": [data_layer_arn, ml_layer_arn],
        "ml_training": [data_layer_arn, ml_layer_arn],
        "visualization": [data_layer_arn, ml_layer_arn],
        "ai_assistant": [data_layer_arn],
    }

    arns = {}
    for func in FUNCTIONS:
        func_layers = layer_map.get(func["dir"], [])
        arn = deploy_function(func, role_arn, account_id, func_layers)
        arns[func["dir"]] = arn

    print("\n" + "=" * 60)
    print("All Lambda functions deployed!")
    print("=" * 60)
    print("\nLambda ARNs:")
    for dir_name, arn in arns.items():
        print(f"  {dir_name}: {arn}")

    # Output ARNs in a format the deploy script can use
    arns_file = os.path.join(PROJECT_ROOT, "lambda-arns.json")
    with open(arns_file, "w") as f:
        json.dump(arns, f, indent=2)
    print(f"\nARNs saved to: {arns_file}")


if __name__ == "__main__":
    main()
