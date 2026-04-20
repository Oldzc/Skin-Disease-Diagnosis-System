# 皮肤疾病初筛系统（毕业设计原型）

## 1. 项目简介
本项目实现了一个“**皮肤图像 + 症状信息 -> 结构化初筛结果**”的多层诊断原型系统，定位为教学/科研用途的辅助筛查工具，不用于临床确诊。

系统目标：
- 提供可演示的完整推理链路（API + 本地智能 + 规则兜底）
- 支持多模型 API 与本地离线推理
- 提供可复现实验与图表产出（适配论文/答辩）

标准输出示例：
```json
{
  "primary_diagnosis": "Eczema",
  "confidence": 0.78,
  "source": "qwen_vl_api | local_hybrid | local_mock",
  "mock_result": false,
  "note": "初步筛查结果，非临床诊断结论",
  "top3_candidates": [
    {"label": "Eczema", "score": 0.78},
    {"label": "Psoriasis", "score": 0.11},
    {"label": "Tinea", "score": 0.06}
  ],
  "decision_trace": {}
}
```

---

## 2. 核心能力

### 2.1 三层推理降级链
1. **大模型 API 层**（优先）  
   支持 `Qwen / OpenAI / Anthropic / Gemini` 多提供商图文推理。
2. **本地智能层（local_hybrid）**  
   图像模型概率 + 文本规则概率 + 类别先验融合：
   `P_final = α·P_image + β·P_text + γ·P_prior`
3. **本地规则层（local_mock）**  
   当 API 和本地模型不可用时，使用确定性规则兜底，保障流程可跑通。

### 2.2 结构化症状输入（8项）
- 病程、部位、瘙痒程度、疼痛程度
- 诱因、皮损形态、是否复发、年龄段

### 2.3 用户系统与本地历史记录
- 登录/注册（`users.json`，带 salt + sha256）
- 历史按用户隔离（`user_histories/<username>.json`）
- 7天自动过期，支持检索与清空

### 2.4 隐私保护
- 图像预处理后剥离 EXIF 元数据
- 用户需勾选知情同意后方可分析
- 历史仅保存结果摘要，不存原始图片

---

## 3. 技术栈
- Python 3.10+（建议 3.10~3.12）
- Streamlit
- PyTorch / torchvision
- scikit-learn / matplotlib / Pillow
- requests

依赖见 [requirements.txt](D:\SCUT\毕业设计\Code\requirements.txt)。

---

## 4. 环境与安装（Windows）

### 4.1 固定 Python 环境（本项目当前约定）
```powershell
$env:PROJECT_PY="D:\anaconda3\envs\env_disease_detect_1\python.exe"
& $env:PROJECT_PY --version
```

若你未设置 `PROJECT_PY`，请直接使用完整路径执行命令：
```powershell
& "D:\anaconda3\envs\env_disease_detect_1\python.exe" -m pip install -r requirements.txt
```

### 4.2 安装依赖
```powershell
& $env:PROJECT_PY -m pip install -r requirements.txt
```

---

## 5. 数据集目录
默认目录：
```text
Dataset/archive/SkinDisease/
├── train/
└── test/
```

当前标签为 22 类（目录名即标签）：  
`Acne, Actinic_Keratosis, Benign_tumors, Bullous, Candidiasis, DrugEruption, Eczema, Infestations_Bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrh_Keratoses, SkinCancer, Sun_Sunlight_Damage, Tinea, Unknown_Normal, Vascular_Tumors, Vasculitis, Vitiligo, Warts`

---

## 6. 运行应用

### 6.1 设置 API Key（四选一）
```powershell
# Qwen
$env:QWEN_API_KEY="your_key"

# OpenAI
$env:OPENAI_API_KEY="your_key"

# Anthropic
$env:ANTHROPIC_API_KEY="your_key"

# Gemini
$env:GOOGLE_API_KEY="your_key"

# 可选：默认提供商
$env:PREFERRED_PROVIDER="qwen"   # qwen / openai / anthropic / gemini
```

说明：当前代码默认读取**系统环境变量**（`os.getenv`），不会自动加载 `.env` 文件。

### 6.2 本地模型配置（可选）
```powershell
$env:LOCAL_MODEL_DIR="artifacts/multi_model_compare"
$env:LOCAL_MODEL_ARCH="efficientnet_b0"   # mobilenet_v3_small / resnet18 / efficientnet_b0 / auto
```

### 6.3 启动
```powershell
& $env:PROJECT_PY -m streamlit run app.py
```

浏览器访问：`http://localhost:8501`

---

## 7. 本地模型训练与评测

### 7.1 训练（示例：EfficientNet-B0）
```powershell
& $env:PROJECT_PY scripts/train_local_model.py `
  --dataset-root Dataset/archive/SkinDisease `
  --artifacts-dir artifacts/multi_model_compare/efficientnet_b0 `
  --arch efficientnet_b0 `
  --epochs 30 `
  --batch-size 32 `
  --lr 3e-4 `
  --weight-decay 1e-4 `
  --freeze-epochs 5 `
  --imbalance-strategy class_weight `
  --early-stop-patience 6 `
  --use-amp
```

支持架构：
- `mobilenet_v3_small`
- `resnet18`
- `efficientnet_b0`

训练产物（每个模型目录）：
- `local_model.pkl`
- `label_map.json`
- `metrics.json`
- `train_manifest.csv`
- `test_manifest.csv`

### 7.2 本地方法评测
```powershell
& $env:PROJECT_PY scripts/evaluate_local_methods.py `
  --dataset-root Dataset/archive/SkinDisease `
  --artifacts-dir artifacts/multi_model_compare/efficientnet_b0 `
  --max-per-class 40 `
  --symptom-mode label_hint `
  --output artifacts/multi_model_compare/efficientnet_b0/local_eval_report.json
```

对比三种方法：
- `old_mock`
- `image_only`
- `image_text_fusion`

### 7.3 单模型可视化
```powershell
& $env:PROJECT_PY scripts/plot_training_report.py `
  --artifacts-dir artifacts/multi_model_compare/efficientnet_b0 `
  --output-dir artifacts/figures
```

### 7.4 三模型结果可视化
```powershell
& $env:PROJECT_PY scripts/plot_multi_model_report.py `
  --multi-artifacts-dir artifacts/multi_model_compare `
  --output-dir artifacts/multi_model_compare/figures
```

### 7.5 三模型训练曲线（Loss/Acc/F1）
```powershell
& $env:PROJECT_PY scripts/plot_multi_model_training_curves.py `
  --multi-artifacts-dir artifacts/multi_model_compare `
  --output-dir artifacts/figures
```

---

## 8. 实验体系（Exp1 ~ Exp7）

总汇总文件：  
`artifacts/experiment_summary.csv`

| 编号 | 实验 | 目的 |
|---|---|---|
| Exp1 | 模型选型对比（3 backbone） | 精度/大小/延时对比 |
| Exp2 | 多模态输入对比 | 验证文本信息增益 |
| Exp3 | 融合权重扫描 | 验证 `α,β,γ` 对性能影响 |
| Exp4 | Prompt & JSON约束 | 验证结构化输出稳定性 |
| Exp5 | 鲁棒性（故障注入） | 验证降级路由正确性 |
| Exp6 | 消融实验 | 验证关键模块贡献 |
| Exp7 | 外部泛化（HAM10000） | 验证跨数据集表现 |

### 8.1 运行 Exp2~Exp5
```powershell
& $env:PROJECT_PY scripts/run_experiment_suite.py `
  --dataset-root Dataset/archive/SkinDisease `
  --artifacts-dir artifacts/multi_model_compare/efficientnet_b0 `
  --output-dir artifacts/experiments `
  --max-per-class 5 `
  --seed 42
```

### 8.2 绘制 Exp1~Exp7 论文图
```powershell
& $env:PROJECT_PY scripts/plot_experiment_summary.py `
  --summary-csv artifacts/experiment_summary.csv `
  --output-dir artifacts/figures `
  --dpi 180
```

### 8.3 Colab 实验 Notebook
- `notebooks/colab_train_local_model.ipynb`
- `notebooks/colab_fusion_weight_sweep.ipynb`
- `notebooks/colab_external_generalization_ham10000.ipynb`

---

## 9. 当前实验结果摘要（基于已有产物）

- **Exp1（模型选型）**：`efficientnet_b0` 综合最佳（Top-1=0.7620，Macro-F1=0.7337）
- **Exp2（多模态）**：`image_text_fusion` 相比 `image_only` 提升  
  Top-1 `+0.0091`，Top-3 `+0.0727`，Macro-F1 `+0.0103`
- **Exp3（融合权重）**：多组权重达到 Top-1=0.7545；`a=0.5,b=0.3,g=0.2` 延时显著异常（209.46ms）
- **Exp4（约束实验）**：`raw_prompt + json_constraint` 出现结构失败（Top-1/Macro-F1=0）
- **Exp5（鲁棒性）**：故障场景 B/C 路由正确率=1.0
- **Exp6（消融）**：去文本输入、去降级机制均明显降低性能
- **Exp7（外部泛化）**：HAM10000 上性能下降明显，但系统可用性仍高（success_rate≈1）

---

## 10. 项目结构

```text
Code/
├── app.py
├── core/
│   ├── inference.py
│   ├── local_hybrid.py
│   ├── local_model.py
│   └── mock_engine.py
├── scripts/
│   ├── train_local_model.py
│   ├── evaluate_local_methods.py
│   ├── run_experiment_suite.py
│   ├── plot_experiment_suite.py
│   ├── plot_experiment_summary.py
│   ├── plot_training_report.py
│   ├── plot_multi_model_report.py
│   ├── plot_multi_model_training_curves.py
│   └── smoke_test_qwen_api.py
├── notebooks/
│   ├── colab_train_local_model.ipynb
│   ├── colab_fusion_weight_sweep.ipynb
│   └── colab_external_generalization_ham10000.ipynb
├── artifacts/
│   ├── experiment_summary.csv
│   ├── figures/
│   ├── multi_model_compare/
│   ├── fusion_weight_sweep/
│   └── external_generalization/
├── Dataset/
│   └── archive/SkinDisease/
├── users.json
└── user_histories/
```

---

## 11. 常见问题（FAQ）

### Q1. PowerShell 报错：`& $env:PROJECT_PY ...` 无效对象
原因：`PROJECT_PY` 未设置。  
解决：
```powershell
$env:PROJECT_PY="D:\anaconda3\envs\env_disease_detect_1\python.exe"
& $env:PROJECT_PY --version
```
或者直接用完整路径执行命令。

### Q2. 明明配了 API Key，却显示 missing
请确认在**当前终端会话**内设置，并用下面命令检查：
```powershell
echo $env:QWEN_API_KEY
```

### Q3. Qwen API 报图像尺寸错误（宽/高过小）
系统主流程已做 512x512 预处理；如果你单独调 API，请确保图像尺寸 > 10x10。

### Q4. API 不稳定时怎么办
系统会自动降级到 `local_hybrid`，再降到 `local_mock`，确保可演示。

### Q5. 如何快速验证 Qwen API 连通性
```powershell
& $env:PROJECT_PY scripts/smoke_test_qwen_api.py
```

---

## 12. 合规声明
- 本项目仅用于教学与科研演示，不构成医疗建议。
- 输出结果为“辅助筛查”，不能替代医生诊断。
- 使用外部 API 时请遵守对应平台的服务条款与数据规范。

---

## 13. 版本说明（简）
- **v2.2（当前）**：完成 Exp1~Exp7 实验整合、统一实验图表流程、三模型训练曲线输出、登录与用户隔离历史记录。
- **v2.0**：结构化症状输入、隐私同意、三级降级链、local_hybrid 融合推理。
- **v1.0**：最小可运行演示链路（图片+文本->结构化结果）。
