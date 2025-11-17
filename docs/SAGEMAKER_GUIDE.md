# SageMaker AI Studio 사용 가이드

AWS SageMaker AI Studio에서 Qwen3-VL-2B 모델을 사용하여 데이팅 프로필 매칭 시스템을 구축하는 방법입니다.

**데이터 규모:** 3,300개 이미지 (실제 100개 + 증강 3,200개)

## 목차

1. [SageMaker Studio 설정](#sagemaker-studio-설정)
2. [프로젝트 설정](#프로젝트-설정)
3. [데이터 준비](#데이터-준비)
4. [모델 학습](#모델-학습)
5. [모델 배포 (선택)](#모델-배포-선택)
6. [비용 최적화](#비용-최적화)

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

# 프로젝트 클론 (또는 직접 업로드)
git clone <your-repo-url> dating-profile-matcher
cd dating-profile-matcher

# 의존성 설치
pip install -r requirements.txt

# 디렉토리 구조 생성
mkdir -p data/{raw/profiles,raw/augmented,processed/{train,val}}
mkdir -p models/{checkpoints,saved_models}
mkdir -p logs
```

### 3. 환경 변수 설정 (선택사항)

```bash
# .env 파일 생성
cat > .env << EOF
# AWS 설정
AWS_REGION=ap-southeast-2
S3_BUCKET=your-bucket-name

# SageMaker 설정
SAGEMAKER_ROLE=arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole

# 모델 설정
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
EMBEDDING_DIM=512

# HuggingFace Token (필요시)
HF_TOKEN=your_huggingface_token
EOF
```

## 데이터 준비

### 1. 데이터 구조 (현재 프로젝트)

```
총 3,300개 이미지:
├── 실제 사용자 이미지: 100개 → data/raw/profiles/
│   └── 파일명 형식: user_001_1.jpg, user_001_2.jpg
└── 증강 이미지: 3,200개 → data/raw/augmented/
    └── 파일명 형식: gen_0001.jpg, gen_0002.jpg
```

### 2. 데이터 업로드 방법

#### 옵션 A: SageMaker Studio 파일 브라우저

1. 좌측 파일 브라우저 열기
2. `data/raw/profiles/` 폴더로 이동
3. 실제 이미지 100개 드래그 앤 드롭
4. `data/raw/augmented/` 폴더로 이동
5. 증강 이미지 3,200개 업로드

#### 옵션 B: AWS CLI (대용량 권장)

```bash
# 로컬에서 S3로 업로드
aws s3 sync local_data/profiles/ s3://your-bucket/dating-matcher/profiles/
aws s3 sync local_data/augmented/ s3://your-bucket/dating-matcher/augmented/

# SageMaker에서 S3에서 다운로드
aws s3 sync s3://your-bucket/dating-matcher/profiles/ ~/SageMaker/dating-profile-matcher/data/raw/profiles/
aws s3 sync s3://your-bucket/dating-matcher/augmented/ ~/SageMaker/dating-profile-matcher/data/raw/augmented/
```

### 3. 메타데이터 생성

GETTING_STARTED.md의 `create_metadata.py` 스크립트 참고

## 모델 학습

### 1. Jupyter Notebook에서 모델 로드 테스트

```python
# test_model.ipynb
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 모델 로드
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

print("모델 로드 성공!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. SageMaker Training Job 시작

`notebooks/sagemaker_training.ipynb` 열기:

```python
import sagemaker
from sagemaker.pytorch import PyTorch

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Training Job 설정
estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='../src/training',
    dependencies=['../src', '../configs'],
    role=role,
    instance_type='ml.g5.xlarge',
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'config': 'configs/config.yaml',
        'num-epochs': 15,
        'batch-size': 16,
        'learning-rate': 1e-4
    },
    use_spot_instances=True,
    max_run=3*60*60,  # 3시간
    max_wait=3*60*60,
    checkpoint_s3_uri=f"s3://{sess.default_bucket()}/dating-matcher/checkpoints"
)

# 학습 시작
estimator.fit()
```

**예상 학습 시간:** 2-3시간 (3,300개 이미지, 15 epochs)

### 3. 학습 모니터링

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

### 4. 학습 완료 후 모델 다운로드

```python
# 모델 아티팩트 다운로드
import boto3

s3 = boto3.client('s3')
s3.download_file(
    'your-bucket',
    'dating-matcher/output/model.tar.gz',
    'models/saved_models/model.tar.gz'
)

# 압축 해제
import tarfile
with tarfile.open('models/saved_models/model.tar.gz', 'r:gz') as tar:
    tar.extractall('models/saved_models/')
```

## 모델 배포 (선택)

프로젝트 범위에서는 배포가 필수가 아니지만, 실제 서비스 적용을 원한다면 다음 방법을 사용할 수 있습니다.

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
| ml.g5.2xlarge | 8 | 1 (A10G 24GB) | 32 GB | ~$2.03 | 본 프로젝트 권장 |
| ml.g5.4xlarge | 16 | 1 (A10G 24GB) | 64 GB | ~$3.26 | 대규모 데이터 |

**현재 프로젝트 (3,300개 이미지) 권장:**

- **ml.g5.xlarge** + Spot 인스턴스
- 예상 비용: 약 $0.42/시간 × 3시간 = **~$1.26**

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

**증상:** CUDA out of memory 에러

**해결방법:**

```yaml
# config.yaml 수정
training:
  batch_size: 8  # 16에서 8로 줄임
```

또는 Gradient Accumulation:

```python
# train_sagemaker.py에서
gradient_accumulation_steps = 4  # Effective batch = 8 × 4 = 32
```

### HuggingFace 모델 다운로드 실패

**증상:** 401 Unauthorized 또는 다운로드 타임아웃

**해결방법:**

```python
# HuggingFace 토큰 설정
from huggingface_hub import login
login(token="your_huggingface_token")

# 또는 환경 변수
export HF_TOKEN=your_token
```

### Spot 인스턴스 중단

**증상:** Training job이 중간에 멈춤

**해결방법:**

체크포인트에서 자동 재개:

```python
# 중단된 job의 체크포인트 경로 확인
checkpoint_s3_uri = 's3://your-bucket/dating-matcher/checkpoints'

# 동일한 설정으로 재시작 (자동으로 last.pth 로드)
estimator.fit()
```

### 학습 속도가 너무 느림

**해결방법:**

1. **Mixed Precision Training** (FP16):
   ```python
   # train_sagemaker.py에서
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

2. **Vision Encoder Freeze**:
   ```yaml
   # config.yaml
   model:
     freeze_vision_encoder: true  # Stage 1
   ```

3. **DataLoader Workers 증가**:
   ```python
   train_loader = DataLoader(
       dataset, 
       batch_size=16, 
       num_workers=4  # CPU 코어 수에 맞게
   )
   ```

## 모범 사례

### 1. 실험 관리

SageMaker Experiments로 여러 실험 추적:

```python
from sagemaker.experiments import Run

with Run(experiment_name='dating-matcher-experiments') as run:
    run.log_parameters({
        'learning_rate': 5e-5,
        'batch_size': 16,
        'margin': 1.0
    })
    
    estimator.fit({'training': s3_data_uri})
    
    run.log_metric(name='final_loss', value=0.234)
```

### 2. 데이터 버전 관리

S3에 날짜별 버전으로 저장:

```bash
# 데이터 업로드 시 버전 포함
aws s3 sync data/raw/ s3://bucket/dating-matcher/data/v20250117/

# 학습 시 특정 버전 사용
s3_data = f"s3://bucket/dating-matcher/data/v20250117/"
```

### 3. 비용 추적

AWS Cost Explorer로 비용 모니터링:

```python
# 예상 비용 계산
def estimate_cost(instance_type='ml.g5.xlarge', hours=3, use_spot=True):
    prices = {
        'ml.g5.xlarge': 1.41,
        'ml.g5.2xlarge': 2.03
    }
    hourly = prices.get(instance_type, 0)
    if use_spot:
        hourly *= 0.3  # 70% 할인
    return hourly * hours

# 현재 프로젝트 예상 비용
cost = estimate_cost('ml.g5.xlarge', 3, use_spot=True)
print(f"예상 학습 비용: ${cost:.2f}")
# 출력: 예상 학습 비용: $1.27
```

## 다음 단계

### 프로젝트 완료 후

1. **모델 평가**: GETTING_STARTED.md의 평가 섹션 참고
2. **결과 분석**: 임베딩 품질 지표 확인
3. **보고서 작성**: 실험 결과 및 학습 내용 정리

### 향후 확장 (선택사항)

1. **실제 피드백 데이터 수집**: 좋아요/패스 데이터로 재학습
2. **하이퍼파라미터 튜닝**: SageMaker Automatic Model Tuning 활용
3. **프로덕션 배포**: 엔드포인트 배포 및 A/B 테스트
4. **멀티모달 확장**: 텍스트 프로필 정보 통합

## 참고 자료

### AWS 공식 문서

- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker 가격 정보](https://aws.amazon.com/sagemaker/pricing/)
- [PyTorch on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/index.html)

### 프로젝트 문서

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md): 프로젝트 전체 맥락
- [GETTING_STARTED.md](GETTING_STARTED.md): 빠른 시작 가이드
- [ARCHITECTURE.md](ARCHITECTURE.md): 시스템 아키텍처
- [DATA_SPEC.md](DATA_SPEC.md): 데이터 명세

### 모델 관련

- [Qwen-VL Documentation](https://huggingface.co/Qwen)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)
