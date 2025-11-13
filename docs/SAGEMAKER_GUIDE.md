# SageMaker AI Studio 사용 가이드

AWS SageMaker AI Studio에서 Qwen3-VL 모델을 사용하여 데이팅 프로필 매칭 시스템을 구축하는 방법입니다.

## 목차

1. [SageMaker Studio 설정](#sagemaker-studio-설정)
2. [프로젝트 설정](#프로젝트-설정)
3. [데이터 준비](#데이터-준비)
4. [로컬 학습](#로컬-학습)
5. [SageMaker Training Job](#sagemaker-training-job)
6. [모델 배포](#모델-배포)
7. [비용 최적화](#비용-최적화)

## SageMaker Studio 설정

### 1. SageMaker Studio 도메인 생성

```bash
# AWS CLI를 사용한 도메인 생성 (또는 AWS Console 사용)
aws sagemaker create-domain \
    --domain-name dating-matcher-domain \
    --auth-mode IAM \
    --default-user-settings ExecutionRole=arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole \
    --subnet-ids subnet-xxxxx \
    --vpc-id vpc-xxxxx
```

### 2. Studio 접속

1. AWS Console → SageMaker → Domains
2. 생성한 도메인 선택
3. "Launch" → "Studio" 클릭

### 3. 필수 IAM 권한

SageMaker 실행 역할에 다음 권한이 필요합니다:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpoint"
      ],
      "Resource": "*"
    }
  ]
}
```

## 프로젝트 설정

### 1. Studio에서 터미널 열기

SageMaker Studio 좌측 메뉴에서 "File" → "New" → "Terminal"

### 2. 프로젝트 클론 및 설정

```bash
# 홈 디렉토리로 이동
cd ~/SageMaker

# 프로젝트 디렉토리가 이미 있다면 해당 위치로 이동
cd dating-profile-matcher

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# AWS 설정
AWS_REGION=ap-southeast-2
S3_BUCKET=your-bucket-name

# SageMaker 설정
SAGEMAKER_ROLE=arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole

# 모델 설정
MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct-FP8
EMBEDDING_DIM=512
EOF
```

## 데이터 준비

### 1. 로컬 데이터 구조 생성

```bash
mkdir -p ~/SageMaker/dating-matcher/data/{raw,processed}
```

### 2. 데이터 업로드

**옵션 A: 로컬 파일에서 업로드**

```bash
# Studio 파일 브라우저에서 직접 업로드
# 또는 S3에서 다운로드
aws s3 sync s3://your-bucket/dating-data ~/SageMaker/dating-matcher/data/raw/
```

**옵션 B: S3에 직접 업로드**

```bash
# 로컬에서 S3로 업로드
aws s3 sync data/raw/ s3://your-bucket/dating-matcher/data/raw/

# 메타데이터 CSV 업로드
aws s3 cp data/raw/metadata.csv s3://your-bucket/dating-matcher/data/raw/
```

### 3. 데이터 전처리

```bash
# SageMaker Studio 터미널에서 실행
python src/data/preprocessing.py \
    --input_dir ~/SageMaker/dating-matcher/data/raw/profiles \
    --output_dir ~/SageMaker/dating-matcher/data/processed \
    --metadata_csv ~/SageMaker/dating-matcher/data/raw/metadata.csv \
    --output_metadata_csv ~/SageMaker/dating-matcher/data/processed/metadata.csv \
    --image_size 224
```

## 로컬 학습 (Studio 인스턴스에서)

### 1. 소규모 데이터로 테스트

```python
# test_local.py
import torch
from src.models.qwen_backbone import Qwen3VLFeatureExtractor

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 모델 로드 테스트
model = Qwen3VLFeatureExtractor(
    model_name="Qwen/Qwen3-VL-2B-Instruct-FP8",
    embedding_dim=512,
    device=device
)

print("모델 로드 성공!")
```

### 2. 로컬 학습 실행

```bash
# 작은 배치로 빠른 테스트
python src/training/train_sagemaker.py \
    --config configs/config_sagemaker.yaml \
    --batch-size 4 \
    --num-epochs 2
```

**주의사항:**
- SageMaker Studio 기본 인스턴스는 GPU가 없을 수 있습니다
- 본격 학습은 Training Job 사용 권장
- 로컬 테스트는 코드 검증 목적으로만 사용

## SageMaker Training Job

### 1. Jupyter Notebook으로 Training Job 시작

`notebooks/sagemaker_training.ipynb` 열기:

```python
# 주요 설정
estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='../src/training',
    instance_type='ml.g5.xlarge',  # GPU 인스턴스
    instance_count=1,
    framework_version='2.1.0',
    hyperparameters={
        'learning-rate': 5e-5,
        'batch-size': 16,
        'num-epochs': 30
    }
)

# 학습 시작
estimator.fit({'training': s3_data_uri})
```

### 2. Python 스크립트로 Training Job 시작

```python
# launch_training.py
import sagemaker
from sagemaker.pytorch import PyTorch

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='src/training',
    dependencies=['src', 'configs'],
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'config': 'configs/config_sagemaker.yaml'
    },
    use_spot_instances=True,
    max_run=24*60*60,
    checkpoint_s3_uri=f"s3://{sess.default_bucket()}/dating-matcher/checkpoints"
)

s3_data = f"s3://{sess.default_bucket()}/dating-matcher/data/processed"
estimator.fit({'training': s3_data})
```

```bash
python launch_training.py
```

### 3. Training Job 모니터링

**AWS Console:**
- SageMaker → Training → Training jobs
- 실행 중인 job 선택
- CloudWatch logs 확인

**Python SDK:**

```python
# 학습 상태 확인
estimator.latest_training_job.describe()

# 로그 실시간 확인
estimator.logs()
```

### 4. 체크포인트 및 모델 저장

체크포인트는 자동으로 S3에 저장됩니다:

```
s3://your-bucket/dating-matcher/checkpoints/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
└── last.pth

s3://your-bucket/dating-matcher/output/
└── model.tar.gz  # 최종 모델
```

## 모델 배포

### 1. SageMaker 엔드포인트 배포

```python
# 엔드포인트 배포
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.xlarge',
    endpoint_name='dating-matcher-endpoint'
)

# 추론 테스트
import json
from PIL import Image

image = Image.open('test_profile.jpg')
response = predictor.predict(image)
print(response)
```

### 2. 커스텀 추론 스크립트 (선택사항)

```python
# inference.py (src/inference/)
def model_fn(model_dir):
    """모델 로드"""
    model = Qwen3VLFeatureExtractor.load_from_checkpoint(
        os.path.join(model_dir, 'model.pth')
    )
    return model

def input_fn(request_body, content_type):
    """입력 전처리"""
    if content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
        return image
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(data, model):
    """추론 실행"""
    embedding = model.get_embedding(data)
    return embedding

def output_fn(prediction, accept):
    """출력 후처리"""
    if accept == 'application/json':
        return json.dumps({'embedding': prediction.tolist()})
    raise ValueError(f"Unsupported accept type: {accept}")
```

### 3. Auto-scaling 설정

```python
import boto3

client = boto3.client('application-autoscaling')

# Auto-scaling 등록
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=5
)

# Scaling 정책
client.put_scaling_policy(
    PolicyName='dating-matcher-scaling',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 100.0,  # 인스턴스당 100 invocations
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

## 비용 최적화

### 1. Spot 인스턴스 사용

Training Job에서 Spot 인스턴스 사용 시 최대 70% 비용 절감:

```python
estimator = PyTorch(
    # ... 기타 설정
    use_spot_instances=True,
    max_run=24*60*60,  # 최대 실행 시간
    max_wait=24*60*60  # Spot 인스턴스 대기 시간
)
```

### 2. 인스턴스 타입 선택

| 인스턴스 타입 | vCPU | GPU | 메모리 | 시간당 비용 (ap-southeast-2) | 추천 용도 |
|--------------|------|-----|--------|------------------------------|-----------|
| ml.g5.xlarge | 4 | 1 (A10G 24GB) | 16 GB | ~$1.41 | 개발/테스트 |
| ml.g5.2xlarge | 8 | 1 (A10G 24GB) | 32 GB | ~$2.03 | 소규모 학습 |
| ml.g5.4xlarge | 16 | 1 (A10G 24GB) | 64 GB | ~$3.26 | 중규모 학습 |
| ml.g5.12xlarge | 48 | 4 (A10G 24GB) | 192 GB | ~$9.08 | 대규모 학습 |

**권장:**
- 개발/테스트: ml.g5.xlarge + Spot 인스턴스
- 프로덕션 학습: ml.g5.2xlarge + Spot 인스턴스

### 3. Early Stopping

```python
from sagemaker.tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    # ... 기타 설정
    early_stopping_type='Auto'  # 개선이 없으면 자동 중단
)
```

### 4. 엔드포인트 비용 절감

```python
# 사용하지 않을 때는 엔드포인트 삭제
predictor.delete_endpoint()

# 또는 Serverless Inference 사용 (낮은 트래픽일 때)
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=10
)

predictor = model.deploy(
    serverless_inference_config=serverless_config
)
```

### 5. 예상 비용 계산

```python
# 학습 비용 예상
def estimate_training_cost(
    instance_type='ml.g5.xlarge',
    hours=3,
    use_spot=True
):
    prices = {
        'ml.g5.xlarge': 1.41,
        'ml.g5.2xlarge': 2.03
    }

    hourly_rate = prices.get(instance_type, 0)

    if use_spot:
        hourly_rate *= 0.3  # Spot은 약 70% 할인

    total_cost = hourly_rate * hours

    print(f"인스턴스: {instance_type}")
    print(f"예상 시간: {hours}시간")
    print(f"Spot 사용: {use_spot}")
    print(f"예상 비용: ${total_cost:.2f}")

    return total_cost

# 예시
estimate_training_cost('ml.g5.xlarge', 3, use_spot=True)
# 출력: 예상 비용: $1.27
```

## 문제 해결

### GPU 메모리 부족

```yaml
# config_sagemaker.yaml 수정
training:
  batch_size: 8  # 16에서 8로 줄임
  gradient_accumulation_steps: 4  # Effective batch = 32
```

### HuggingFace 모델 다운로드 실패

```python
# 환경 변수 설정
environment = {
    'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
    'HF_HOME': '/tmp/huggingface',
    'HF_TOKEN': 'your_huggingface_token'  # 필요시
}
```

### Spot 인스턴스 중단

체크포인트가 자동으로 S3에 저장되므로 재개 가능:

```python
# 중단된 job에서 재개
checkpoint_s3_uri = 's3://your-bucket/dating-matcher/checkpoints'

estimator = PyTorch(
    # ... 기타 설정
    checkpoint_s3_uri=checkpoint_s3_uri
)

# S3에서 last.pth를 자동으로 로드하고 재개
estimator.fit({'training': s3_data_uri})
```

## 모범 사례

### 1. 실험 관리

SageMaker Experiments 사용:

```python
from sagemaker.experiments import Run

with Run(experiment_name='dating-matcher-experiments') as run:
    run.log_parameters({
        'learning_rate': 5e-5,
        'batch_size': 16
    })

    estimator.fit({'training': s3_data_uri})

    run.log_metric(name='final_loss', value=0.234)
```

### 2. 데이터 버전 관리

```bash
# S3에 날짜별 버전 저장
aws s3 sync data/processed/ s3://bucket/dating-matcher/data/v20240115/
```

### 3. 모델 레지스트리

```python
# 학습된 모델을 모델 레지스트리에 등록
model_package = estimator.register(
    content_types=['application/x-image'],
    response_types=['application/json'],
    inference_instances=['ml.g5.xlarge'],
    transform_instances=['ml.g5.xlarge'],
    model_package_group_name='dating-matcher-models',
    approval_status='PendingManualApproval'
)
```

## 다음 단계

1. **A/B 테스트**: 새 모델과 기존 모델 비교
2. **모델 모니터링**: SageMaker Model Monitor 설정
3. **파이프라인 구축**: SageMaker Pipelines로 자동화
4. **멀티모델 엔드포인트**: 여러 버전 동시 서빙

## 참고 자료

- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Qwen-VL Documentation](https://huggingface.co/Qwen)
- [SageMaker 가격 정보](https://aws.amazon.com/sagemaker/pricing/)
