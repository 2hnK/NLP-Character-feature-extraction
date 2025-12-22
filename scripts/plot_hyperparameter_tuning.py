"""
하이퍼파라미터 튜닝 결과 시각화 스크립트
논문 Figure용 막대 그래프 생성
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터
experiments = ['Exp1\n(Initial)', 'Exp2\n(+Dropout)', 'Exp3\n(+Temp 0.1)', 'Exp4\n(Final)']
recall_at_1 = [0.78, 0.97, 1.49, 1.69]
recall_at_5 = [5.71, 4.93, 5.77, 7.85]
recall_at_10 = [10.18, 10.18, 11.28, 14.98]

# 색상 설정
colors = ['#95a5a6', '#95a5a6', '#3498db', '#2ecc71']
edge_colors = ['#7f8c8d', '#7f8c8d', '#2980b9', '#27ae60']

# Figure 설정
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

# 막대 그래프
x = np.arange(len(experiments))
width = 0.6

bars = ax.bar(x, recall_at_1, width, color=colors, edgecolor=edge_colors, linewidth=1.5)

# 값 라벨 추가
for bar, val in zip(bars, recall_at_1):
    height = bar.get_height()
    ax.annotate(f'{val:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')

# 랜덤 기준선
ax.axhline(y=0.65, color='#e74c3c', linestyle='--', linewidth=1.5, label='Random Baseline (0.65%)')

# 축 설정
ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
ax.set_title('Hyperparameter Tuning Results', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=10)
ax.set_ylim(0, 2.2)

# 그리드
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# 범례
ax.legend(loc='upper left', fontsize=10)

# 향상률 화살표 추가
ax.annotate('', xy=(3, 1.69), xytext=(0, 0.78),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2, 
                          connectionstyle='arc3,rad=0.3'))
ax.text(1.5, 1.4, '+117%', fontsize=12, fontweight='bold', color='#27ae60',
        ha='center', va='bottom')

plt.tight_layout()

# 저장
output_path = r'd:\Document_2025\NLP-Character-feature-extraction\paper\figures\hyperparameter_tuning.png'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"✅ 그래프 저장 완료: {output_path}")
print(f"✅ PDF 저장 완료: {output_path.replace('.png', '.pdf')}")

plt.show()
