# Qwen2-VL 모델 수정 완료

## 🔧 수정 내용

Qwen2-VL의 vision encoder 직접 호출 방식에서 **전체 모델의 forward pass 사용** 방식으로 변경했습니다.

### 변경 사항

1. **`extract_vision_features` 메서드**
   - 기존: vision encoder 직접 호출 (`self.model.visual()`)
   - 수정: 전체 모델 forward pass + hidden_states 추출

2. **`forward` 메서드**
   - autocast 방식 업데이트 (FutureWarning 해결)
   - CPU/GPU 자동 감지

3. **hidden_size 설정**
   - 기존: `vision_config.hidden_size` 사용
   - 수정: `config.hidden_size` 사용 (전체 모델)

---

## 🧪 다시 테스트

```bash
python scripts/test_qwen_model.py
```

### ⚠️ CPU 환경 주의사항

**현재 ml.t3.large (CPU 전용) 환경입니다:**
- 모델 로드: 정상 작동
- Forward pass: **매우 느림** (5-10배 이상)
- 예상 소요 시간: 30-60분

**권장:**
1. 일단 테스트 시작
2. TEST 1, 2가 성공하면 → 코드는 정상
3. 실제 학습은 **GPU 인스턴스** (ml.g5.xlarge)로 진행

---

## 🚀 빠른 테스트 (CPU)

전체 테스트가 너무 느리면 간단한 테스트만:

```bash
python -c "
import torch
from src.models.qwen_backbone import Qwen3VLFeatureExtractor
from PIL import Image
import numpy as np

print('Loading model...')
device = 'cpu'  # CPU로 강제
model = Qwen3VLFeatureExtractor(
    model_name='Qwen/Qwen2-VL-2B-Instruct',
    embedding_dim=512,
    freeze_vision_encoder=True,
    device=device
)
print('✓ Model loaded')

print('Testing forward pass...')
dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

with torch.no_grad():
    emb = model([dummy_img])

print(f'✓ Forward pass successful!')
print(f'  Output shape: {emb.shape}')
print(f'  Embedding norm: {emb.norm(dim=1).item():.4f}')
"
```

**예상 결과:**
```
Loading model...
Loading Qwen3-VL model: Qwen/Qwen2-VL-2B-Instruct
Vision encoder frozen
✓ Model loaded
Testing forward pass...
✓ Forward pass successful!
  Output shape: torch.Size([1, 512])
  Embedding norm: 1.0000
```

---

## 📊 다음 단계

### CPU 환경에서 할 수 있는 것
- ✅ 코드 검증
- ✅ 데이터 준비
- ✅ 메타데이터 생성
- ❌ 실제 학습 (너무 느림)

### GPU 인스턴스로 변경 (권장)

**SageMaker Studio에서:**
1. File → Shut Down All
2. 새 노트북/터미널 시작 시
3. Instance type: **ml.g5.xlarge** 선택
4. Start

**비용:**
- ml.g5.xlarge: $1.41/시간
- Spot 인스턴스: $0.42/시간 (70% 절감)

---

**수정 완료: 2025-11-18**
