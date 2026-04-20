# 皮肤疾病初筛演示原型（增强版）

## 项目目标
输入一张皮肤图片 + 结构化症状信息，输出结构化初步筛查结果（辅助筛查，不用于临床确诊）。

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

## Python环境（固定）
后续统一使用：`D:\anaconda3\envs\env_disease_detect_1\python.exe`

```powershell
$env:PROJECT_PY="D:\anaconda3\envs\env_disease_detect_1\python.exe"
& $env:PROJECT_PY --version
```

## 核心功能

### 1. 结构化症状输入
用户通过 8 个结构化字段填写症状信息，替代自由文本输入：
- **病程**：急性（近几天）/ 慢性（数周以上）/ 不确定
- **部位**：面部 / 躯干 / 四肢 / 头皮 / 其他
- **瘙痒程度**：无 / 轻微 / 中度 / 剧烈
- **疼痛程度**：无 / 轻微 / 中度 / 剧烈
- **诱因**：日晒 / 用药后 / 虫咬 / 接触刺激物 / 不明
- **皮损形态**：红斑 / 鳞屑 / 水疱 / 溃疡 / 丘疹 / 色素改变 / 其他
- **是否复发**：首次发作 / 反复发作 / 不确定
- **年龄段**：儿童（<14）/ 青年（14-35）/ 中年（36-59）/ 老年（≥60）

系统自动将表单选项拼接为标准化文本（如 "慢性，面部，剧烈瘙痒，无疼痛，日晒诱因，红斑，反复发作，老年。"），传入推理引擎。

### 2. 多模型 API 支持 + 三级推理降级链

系统支持 4 个大模型提供商，在设置页面一键切换：

| 提供商 | 环境变量 | 默认模型 | 接口类型 |
|--------|----------|----------|----------|
| Qwen-VL（阿里云） | `QWEN_API_KEY` | `qwen-vl-max-latest` | OpenAI-compatible |
| OpenAI（GPT-4o） | `OPENAI_API_KEY` | `gpt-4o` | OpenAI-compatible |
| Anthropic（Claude） | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` | Anthropic Messages API |
| Google Gemini | `GOOGLE_API_KEY` | `gemini-1.5-pro-latest` | Generative Language API |

推理降级顺序：
1. **大模型 API**（配置了 API Key 时优先）：图文联合推理，返回 Top-3 候选
2. **local_hybrid**（智能兜底）：本地融合推理
   - 图像分支：MobileNetV3-Small（预训练 ImageNet，微调分类头）
   - 文本分支：基于规则的关键词/部位/病程/严重度/瘙痒/疼痛/诱因/皮损形态/复发/年龄加权打分
   - 先验分支：基于训练集类别分布
   - 融合公式：`P_final = α·P_image + β·P_text + γ·P_prior`（默认 α=0.6, β=0.35, γ=0.05）
3. **local_mock**（规则兜底）：纯关键词匹配 + 确定性哈希选择（当本地模型工件缺失时）

### 3. 历史记录管理
- **本地存储**：`user_histories/<username>.json` 按用户分别记录推理输入、结果、来源、时间
- **账号隔离**：每个用户仅查看和管理自己的历史记录（未登录使用 `guest` 账号）
- **自动过期**：超过 7 天的记录自动清理
- **侧边栏展示**：可展开卡片列表，显示诊断、置信度、症状、Top-3 候选
- **关键词检索**：支持按诊断名（中英文）、症状文本、文件名模糊搜索
- **一键清空**：清空当前账号历史按钮

### 4. 隐私保护机制
- **EXIF 剥离**：图像预处理时通过 `Image.frombytes()` 重建像素数据，彻底移除 GPS、拍摄时间、设备型号等敏感元数据
- **知情同意**：上传后必须勾选隐私声明复选框才能启动分析，未勾选时"开始分析"按钮禁用
- **数据最小化**：历史记录仅存诊断结果摘要，不存原始图片

### 5. 用户界面
- **主页面**：
  - 左侧 sidebar：用户登录按钮、运行配置按钮、历史记录（搜索 + 可展开列表 + 清空按钮）
  - 顶部：登录按钮位于“运行配置”按钮上方
  - 点击“用户登录”进入独立二级登录页（登录/注册/退出）
  - 中间：上传图片 + 结构化表单 + 隐私声明 + 开始分析 + 结果展示
- **设置页面**（点击按钮进入）：
  - 数据集信息（只读）
  - 本地模型配置（目录路径 + 本地推理网络选择 + 工件检测状态）
  - API 配置：选择提供商 → 填入对应 API Key → 模型名 / Base URL / 超时
  - 各提供商申请地址与说明
  - 配置自动保存到 session state

## 数据目录要求
默认使用：
- `Dataset/archive/SkinDisease/train/*`（22 类，用于训练本地模型）
- `Dataset/archive/SkinDisease/test/*`（22 类，用于评测）

支持的 22 类皮肤疾病：
Acne, Actinic_Keratosis, Benign_tumors, Bullous, Candidiasis, DrugEruption, Eczema, Infestations_Bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrh_Keratoses, SkinCancer, Sun_Sunlight_Damage, Tinea, Unknown_Normal, Vascular_Tumors, Vasculitis, Vitiligo, Warts

## 快速运行

### 1. 安装依赖
```powershell
& $env:PROJECT_PY -m pip install -r requirements.txt
```

依赖包：
- `streamlit>=1.30.0` — Web UI 框架
- `requests>=2.31.0` — API 调用
- `pillow>=10.0.0` — 图像处理
- `numpy>=1.24.0` — 数值计算
- `torch>=2.2.0` — 深度学习框架
- `torchvision>=0.17.0` — 图像模型
- `scikit-learn>=1.4.0` — 评测指标
- `joblib>=1.3.0` — 模型序列化

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key（`.env` 已在 `.gitignore` 中，不会上传）：

```bash
cp .env.example .env
# 编辑 .env，填入对应的 API Key
```

或直接设置环境变量（四选一，留空则自动使用本地推理）：

```powershell
# Qwen-VL（阿里云通义千问）
$env:QWEN_API_KEY="sk-xxxxxxxxxxxxxxxx"

# OpenAI（GPT-4o）
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Anthropic（Claude）
$env:ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxx"

# Google Gemini
$env:GOOGLE_API_KEY="AIzaxxxxxxxxxxxxxxxx"

# 指定默认提供商（可选，默认 qwen）
$env:PREFERRED_PROVIDER="openai"   # qwen / openai / anthropic / gemini

# 本地模型目录（默认 artifacts/multi_model_compare）
$env:LOCAL_MODEL_DIR="artifacts/multi_model_compare"

# 本地推理网络默认值（推荐 efficientnet_b0）
$env:LOCAL_MODEL_ARCH="efficientnet_b0"
```

### 3. 启动应用
```powershell
& $env:PROJECT_PY -m streamlit run app.py
```

访问 `http://localhost:8501`，按以下步骤使用：
1. 点击侧边栏"运行配置"，选择模型提供商并填入 API Key
2. 上传皮肤图片（JPG/PNG）
3. 填写 8 个结构化症状字段
4. 勾选隐私声明复选框
5. 点击"开始分析"
6. 查看结果：初步诊断、置信度、Top-3 候选、决策轨迹

## 本地模型训练

### 训练命令
```powershell
& $env:PROJECT_PY scripts/train_local_model.py `
  --dataset-root Dataset/archive/SkinDisease `
  --artifacts-dir artifacts `
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

### 参数说明
- `--dataset-root`：数据集根目录（需包含 train/ 和 test/ 子目录）
- `--artifacts-dir`：输出目录（默认 artifacts）
- `--arch`：模型架构（`mobilenet_v3_small` / `resnet18` / `efficientnet_b0`）
- `--epochs`：训练总轮数（默认 30）
- `--batch-size`：批大小
- `--lr`：学习率（默认 3e-4）
- `--weight-decay`：权重衰减（默认 1e-4）
- `--freeze-epochs`：前 N 轮冻结骨干网络，仅训练分类头（默认 5）
- `--pretrained`：使用 ImageNet 预训练权重（默认开启）
- `--imbalance-strategy`：类别不平衡策略（`class_weight` 或 `focal`）
- `--focal-gamma`：Focal Loss 的 gamma（默认 2.0）
- `--use-weighted-sampler`：可选按类别权重采样（默认关闭）
- `--early-stop-patience`：早停耐心轮数（默认 6，监控 macro_f1）
- `--use-amp`：GPU 混合精度训练（默认开启）
- `--expected-num-classes`：期望类别数校验（默认 22）

### Google Colab GPU 训练
- 已提供 Notebook：`notebooks/colab_train_local_model.ipynb`
- Notebook 按 8 段固定流程执行：环境准备 -> 挂载 Drive -> 数据检查 -> 训练配置 -> 训练 -> 评测 -> 导出工件 -> 下载/同步
- Colab 训练完成后，将导出的 `local_model.pkl`、`label_map.json`、`metrics.json` 放回本地 `artifacts/` 即可被 `local_hybrid` 直接加载。
- Notebook 现支持三模型自动对比：`mobilenet_v3_small`、`resnet18`、`efficientnet_b0`，并输出 `compare_summary.csv/json`、`compare_top1_macrof1.png`、`compare_size_latency.png`。
- 新增融合权重扫描 Notebook：`notebooks/colab_fusion_weight_sweep.ipynb`（用于测试 `P_final = α·P_image + β·P_text + γ·P_prior` 的参数组合并导出 CSV/JSON/图表）。

### 训练产物
训练完成后在 `artifacts/` 目录生成：
- `local_model.pkl` — PyTorch 模型权重（~6MB）
- `label_map.json` — 22 类标签映射
- `metrics.json` — 训练指标（含 `best_epoch`、`best_macro_f1`、`class_weights`、`per_class_recall`、混淆矩阵等）
- `train_manifest.csv` — 训练集清单（路径、标签、索引）
- `test_manifest.csv` — 测试集清单

## 本地方法评测

### 评测命令
```powershell
& $env:PROJECT_PY scripts/evaluate_local_methods.py `
  --dataset-root Dataset/archive/SkinDisease `
  --artifacts-dir artifacts/multi_model_compare/efficientnet_b0 `
  --max-per-class 40 `
  --symptom-mode label_hint
```

### 参数说明
- `--max-per-class`：每类最多评测样本数（避免类别不平衡）
- `--symptom-mode`：症状文本生成模式
  - `label_hint`：使用 `SYMP_TEMPLATE` 中的标准症状描述
  - `empty`：空文本（仅图像推理）

### 评测输出
生成 `artifacts/multi_model_compare/efficientnet_b0/local_eval_report.json`，包含三种方法的对比：
1. **old_mock**：纯规则兜底（关键词匹配）
2. **image_only**：仅图像分类（α=1.0, β=0, γ=0）
3. **image_text_fusion**：图像+文本融合（α=0.6, β=0.35, γ=0.05）

评测指标：
- `top1`：Top-1 准确率
- `top3`：Top-3 命中率
- `macro_f1`：宏平均 F1 分数
- `confusion_matrix`：混淆矩阵

### 结果可视化（单独脚本）
```powershell
& $env:PROJECT_PY scripts/plot_training_report.py `
  --artifacts-dir artifacts `
  --output-dir artifacts/figures
```

输出图像：
- `artifacts/figures/training_curves.png`
- `artifacts/figures/method_comparison.png`（若存在 `local_eval_report.json`）
- `artifacts/figures/per_class_recall.png`

### 三模型对比结果可视化（单独脚本）
当你有 `skin_disease_artifacts_export_multi`（包含 `mobilenet_v3_small` / `resnet18` / `efficientnet_b0` 三个子目录）时，
建议放入：`artifacts/multi_model_compare/`，然后运行：

```powershell
& $env:PROJECT_PY scripts/plot_multi_model_report.py `
  --multi-artifacts-dir artifacts/multi_model_compare `
  --output-dir artifacts/multi_model_compare/figures
```

输出：
- `artifacts/multi_model_compare/compare_summary.json`
- `artifacts/multi_model_compare/compare_summary.csv`
- `artifacts/multi_model_compare/figures/compare_accuracy.png`
- `artifacts/multi_model_compare/figures/compare_efficiency.png`
- `artifacts/multi_model_compare/figures/compare_tradeoff.png`

## 实验套件复现（第2-5项）

说明：第1项（三模型选型对比）直接引用已完成结果  
`artifacts/multi_model_compare/compare_summary.csv`，不重复训练。

### 一键运行实验（输出 CSV + JSON）
```powershell
& $env:PROJECT_PY scripts/run_experiment_suite.py `
  --dataset-root Dataset/archive/SkinDisease `
  --artifacts-dir artifacts/multi_model_compare/efficientnet_b0 `
  --output-dir artifacts/experiments `
  --max-per-class 5 `
  --seed 42
```

输出文件：
- `artifacts/experiments/exp2_multimodal.csv/json`
- `artifacts/experiments/exp3_prompt_json.csv/json`
- `artifacts/experiments/exp4_robustness.csv/json`
- `artifacts/experiments/exp5_ablation.csv/json`
- `artifacts/experiments/experiment_summary.csv/json`

若未设置 `QWEN_API_KEY`，API相关实验（第3/4/5项）会自动跳过，并在结果中写入 `skipped_reason`。

### 生成实验图（第2-5项）
```powershell
& $env:PROJECT_PY scripts/plot_experiment_suite.py `
  --experiments-dir artifacts/experiments `
  --output-dir artifacts/experiments
```

输出图像：
- `artifacts/experiments/exp2_multimodal_bar.png`
- `artifacts/experiments/exp3_prompt_json_quality.png`
- `artifacts/experiments/exp4_robustness_route.png`
- `artifacts/experiments/exp5_ablation_impact.png`

## 项目结构

```
Code/
├── app.py                      # Streamlit 主应用（增强版）
├── users.json                  # 本地用户信息（自动生成）
├── user_histories/             # 按用户隔离的历史记录（自动生成，7天过期）
├── requirements.txt            # Python 依赖
├── core/                       # 核心推理模块
│   ├── inference.py            #   统一推理入口（三级降级链）
│   ├── local_hybrid.py         #   本地融合推理（图像+文本+先验）
│   ├── mock_engine.py          #   本地规则兜底引擎
│   └── local_model.py          #   模型构建 + 数据增强
├── scripts/
│   ├── train_local_model.py      # 本地模型训练脚本
│   ├── evaluate_local_methods.py # 三种方法对比评测
│   ├── run_experiment_suite.py   # 第2-5项实验一键编排
│   ├── plot_experiment_suite.py  # 第2-5项实验图表输出
│   ├── plot_training_report.py   # 单模型训练/评测图
│   └── plot_multi_model_report.py # 三模型对比图
├── notebooks/
│   ├── colab_train_local_model.ipynb   # Colab GPU 训练流程
│   └── colab_fusion_weight_sweep.ipynb # Colab 融合权重扫描实验
├── artifacts/                  # 训练产物（需先训练生成）
│   ├── local_model.pkl         #   PyTorch 模型权重
│   ├── label_map.json          #   标签映射
│   ├── metrics.json            #   训练指标
│   ├── train_manifest.csv      #   训练集清单
│   └── test_manifest.csv       #   测试集清单
└── Dataset/                    # 数据集（需自行准备）
    └── archive/SkinDisease/
        ├── train/              #   训练集（22 类子目录）
        └── test/               #   测试集（22 类子目录）
```

## 技术细节

### 图像预处理流程
1. 读取上传图片（支持 JPG/PNG）
2. EXIF 旋转修正（`ImageOps.exif_transpose`）
3. RGB 转换（统一色彩空间）
4. 归一化到 512×512（`ImageOps.fit` + LANCZOS 重采样）
5. **EXIF 剥离**：通过 `Image.frombytes()` 重建像素数据，移除所有元数据
6. 保存为 JPEG（quality=95）

### 文本规则引擎
`core/local_hybrid.py` 中的 `text_probability()` 函数实现了多维度规则打分：
- **关键词匹配**：22 类疾病各有专属关键词词典（中英文），命中后加权
- **否定词检测**：识别"无"、"没有"等否定词，反向扣分
- **部位匹配**：面部/躯干/四肢/头皮，不同疾病有不同部位偏好
- **病程匹配**：急性/慢性，如 DrugEruption 偏急性，Psoriasis 偏慢性
- **严重度匹配**：轻度/中度/重度
- **瘙痒程度**：无/轻微/中度/剧烈，Eczema、Tinea 等瘙痒相关疾病加权
- **疼痛程度**：无/轻微/中度/剧烈，Bullous、SkinCancer 等疼痛相关疾病加权
- **诱因匹配**：日晒/用药/虫咬/接触刺激物，如 Sun_Sunlight_Damage 对日晒加权
- **皮损形态**：红斑/鳞屑/水疱/溃疡/丘疹/色素改变，匹配典型形态
- **复发情况**：首次/反复，Eczema、Psoriasis 等慢性复发疾病加权
- **年龄段**：儿童/青年/中年/老年，如 Acne 偏青年，Seborrh_Keratoses 偏老年

最终通过 softmax 归一化为概率分布。

### 融合策略
```python
P_final = α * P_image + β * P_text + γ * P_prior
```
- `P_image`：本地图像分类模型输出（MobileNetV3-Small）
- `P_text`：文本规则引擎输出（多维度加权打分）
- `P_prior`：训练集类别分布先验
- 默认权重：α=0.6（图像主导），β=0.35（文本辅助），γ=0.05（先验平滑）
- 若症状文本为空，自动调整为 α=0.9, β=0.05, γ=0.05

## 注意事项

### 隐私与合规
- **非医疗器械**：本项目为教学与科研原型，不构成医疗诊断建议，不得用于临床决策
- **数据去标识化**：图像预处理时自动剥离 EXIF 元数据（GPS、时间、设备信息）
- **知情同意**：用户必须勾选隐私声明才能提交分析
- **数据留存**：历史记录仅存诊断结果摘要，不存原始图片，7 天自动过期

### API 调用限制
- Qwen-VL API 可能触发内容审核（`data_inspection_failed`），系统会自动降级到本地推理
- 建议设置合理的超时时间（默认 40 秒）
- API Key 留空时自动跳过 API 调用，直接使用本地推理

### 性能优化
- 本地模型推理速度：~100ms/张（CPU），~20ms/张（GPU）
- 图像预处理：~50ms/张
- 文本规则引擎：<1ms
- 建议使用 GPU 加速（自动检测 CUDA）

### 已知限制
- 仅支持 22 类皮肤疾病，不覆盖所有皮肤病
- 本地模型在小样本类别上准确率较低（如 Lupus、Vitiligo）
- 文本规则引擎基于专家经验，可能存在偏差
- 不支持多病灶同时诊断

## 常见问题

**Q: 如何提高诊断准确率？**
A: 
1. 确保图片清晰、光线充足、病灶居中
2. 尽量填写完整的结构化症状信息
3. 使用大模型 API（在设置页面配置 API Key）
4. 增加训练数据并重新训练本地模型

**Q: 历史记录存在哪里？**
A: `user_histories/<username>.json` 文件，按用户隔离存储；未登录用户使用 `guest.json`。超过 7 天会自动清理。

**Q: 如何导出历史记录？**
A: 直接复制 `user_histories/<username>.json` 文件，或在侧边栏使用"清空当前账号历史"前手动备份

**Q: 如何更换本地模型？**
A: 
1. 重新训练：`python scripts/train_local_model.py --arch efficientnet_b0 --epochs 30 --freeze-epochs 5 --imbalance-strategy class_weight`
2. 替换 `artifacts/multi_model_compare/efficientnet_b0/local_model.pkl` 和 `artifacts/multi_model_compare/efficientnet_b0/label_map.json`
3. 重启应用

**Q: 如何在页面里切换本地推理网络（MobileNet/ResNet/EfficientNet）？**
A:
1. 点击侧边栏"⚙️ 运行配置"。
2. 在"本地模型配置"里设置"本地模型目录"（例如 `artifacts/multi_model_compare`）。
3. 在"本地推理网络"下拉框选择：`MobileNetV3-Small` / `ResNet18` / `EfficientNet-B0` / `自动`。
4. 页面会显示"当前生效模型目录"；若该目录下工件缺失会自动回退 `local_mock`。

**Q: 如何切换大模型提供商？**
A: 点击侧边栏"⚙️ 运行配置"，在"模型提供商"下拉框中选择，填入对应 API Key 即可

**Q: 如何使用第三方 OpenAI 中转服务？**
A: 选择 OpenAI 提供商，将 Base URL 改为中转服务地址（如 `https://api.example.com/v1`），填入中转服务的 API Key

**Q: 如何禁用 API 调用，只用本地推理？**
A: 在设置页面清空 API Key 输入框，或不设置任何 API Key 环境变量

**Q: 如何查看决策轨迹？**
A: 结果页面底部展开"决策轨迹（decision_trace）"，包含：
- 融合权重（α, β, γ）
- 文本匹配信号（matched_signals）
- 否定信号（negated_signals）
- 图像模型信息

## 开源上传指南

上传到 GitHub / Gitee 等开源平台前，请确认以下检查项：

### 必须排除的文件（已在 `.gitignore` 中配置）
- `.env` — 包含真实 API Key
- `users.json` — 本地用户信息
- `user_histories/` — 按用户隔离的历史记录
- `artifacts/multi_model_compare/*/local_model.pkl` — 模型权重文件
- `artifacts/multi_model_compare/*/train_manifest.csv` / `test_manifest.csv` — 数据集路径清单
- `Dataset/` — 数据集目录（体积大）
- `.vscode/` / `.claude/` — 本地 IDE 配置

### 可以上传的文件
- 所有 `.py` 源代码
- `artifacts/multi_model_compare/*/label_map.json` — 标签映射（无隐私）
- `artifacts/multi_model_compare/*/metrics.json` — 训练指标（无隐私）
- `.env.example` — API Key 配置模板
- `requirements.txt` — 依赖列表
- `README.md` — 项目文档
- `.gitignore` — 排除规则

### 初始化 Git 仓库并上传

```bash
# 1. 初始化仓库
git init
git add .
git commit -m "Initial commit: skin disease screening prototype"

# 2. 关联远程仓库（以 GitHub 为例）
git remote add origin https://github.com/your-username/your-repo.git
git branch -M main
git push -u origin main
```

### 上传前最终检查

```bash
# 确认 .env 不在暂存区
git status

# 确认 users.json / user_histories 不在暂存区（应显示为 ignored）
git check-ignore -v users.json
git check-ignore -v user_histories/test.json

# 查看将要上传的文件列表
git ls-files
```

## 更新日志

### v2.1（当前版本）
- ✨ 新增多模型 API 支持（OpenAI GPT-4o / Anthropic Claude / Google Gemini）
- ✨ 新增 `.gitignore` 和 `.env.example`，支持安全开源上传
- 🔧 重构推理引擎为统一 `infer_with_provider()` 接口
- 🔧 设置页面新增提供商选择和各平台申请说明
- ✨ 新增用户登录/注册与按用户隔离历史记录（`users.json` + `user_histories/`）

### v2.0
- ✨ 新增结构化症状输入（8 个字段）
- ✨ 新增历史记录管理（本地存储 + 检索 + 7 天过期）
- ✨ 新增隐私保护机制（EXIF 剥离 + 知情同意）
- ✨ 新增设置页面（独立配置界面）
- 🔧 优化文本规则引擎（新增瘙痒/疼痛/诱因/皮损形态/复发/年龄维度）
- 🔧 优化 UI 布局（sidebar 历史记录 + 主页面输入/结果）
- 🐛 修复 EXIF 元数据泄露问题

### v1.0
- 基础功能：图片上传 + 自由文本输入 + 三级推理降级
- 本地模型训练 + 评测脚本

## 许可与引用
本项目仅供教学与科研使用，不得用于商业用途或临床诊断。
