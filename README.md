# FedMedDP

FedMedDP 是一个面向医疗影像协同诊断的联邦学习实验项目。项目使用 HAM10000 皮肤镜图像构建良性/恶性二分类任务，模拟多客户端训练，并比较无隐私、高斯噪声、拉普拉斯噪声、混合噪声和自适应混合噪声等隐私机制。

项目包含数据整理、联邦训练、差分隐私扰动、指标汇总、成员推断攻击评估和前端实验看板。

## 项目结构

```text
configs/      实验配置文件
docs/         可选部署和说明文档
frontend/     实验看板页面
outputs/      实验输出和汇总结果
scripts/      数据处理、训练、汇总和攻击评估脚本
src/fedmeddp/ 核心训练、模型、数据、指标和隐私代码
requirements.txt
README.md
```

`.gitignore` 已默认排除原始数据、划分后的图片、虚拟环境和模型权重等大文件。建议提交代码、配置、汇总结果和必要说明，不要把 `data/raw_ham10000/`、`data/ham10000_binary/train|val|test/`、`outputs/**/final_model.pt` 直接提交到 GitHub。

## 环境准备

建议使用 Linux 或云端 GPU 环境。CPU 也能跑 demo，但完整 HAM10000 实验会很慢。

本项目已验证的主要环境：

```text
Python: 3.10/3.11 系列更推荐
CUDA: 12.4
torch: 2.6.0+cu124
torchvision: 0.21.0+cu124
numpy: 1.26.4
pandas: 2.3.3
scikit-learn: 1.7.2
matplotlib: 3.10.8
Pillow: 12.1.1
PyYAML: 6.0.3
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`requirements.txt` 默认使用 CUDA 12.4 的 PyTorch 轮子。如果你的环境不是 CUDA 12.4，或只想使用 CPU，请先按 https://pytorch.org/get-started/locally/ 安装匹配的 `torch` 和 `torchvision`，再安装其余依赖。

## 准备数据集

下载 HAM10000 数据集，可选择：

1. Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T
2. Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

下载后放成以下结构：

```text
data/raw_ham10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

整理为二分类 ImageFolder 数据集：

```bash
python scripts/prepare_ham10000.py \
  --src data/raw_ham10000 \
  --dst data/ham10000_binary \
  --group-field lesion_id \
  --seed 2026
```

输出结构：

```text
data/ham10000_binary/
  train/benign/
  train/malignant/
  val/benign/
  val/malignant/
  test/benign/
  test/malignant/
  split_manifest.csv
  split_summary.json
```

HAM10000 原始元数据没有显式 `patient_id`，本项目使用 `lesion_id` 做病灶级隔离划分，避免同一病灶同时出现在训练集、验证集和测试集。

训练前建议审计一次数据：

```bash
python scripts/audit_dataset.py --root data/ham10000_binary
```

审计结果会写入：

```text
data/ham10000_binary/data_quality_report.json
```

## 快速运行 Demo

不想先下载 HAM10000 时，可以用 demo 数据检查代码是否能跑通：

```bash
python scripts/make_demo_dataset.py
python scripts/run_experiment.py --config configs/demo.yaml
python scripts/build_dashboard_data.py --experiment-name demo_experiment
python scripts/serve_demo.py --preload-models
```

启动后会自动选择空闲端口并打开浏览器，终端会打印实际访问地址，例如：

```text
答辩演示地址: http://127.0.0.1:8001/
```

实际端口以终端输出为准。如果不想自动打开浏览器，可加 `--no-browser`。

如果只想看静态看板、不需要现场识别，也可以继续用：

```bash
python -m http.server 8000
```

静态服务不包含现场识别接口，答辩现场建议优先使用 `scripts/serve_demo.py`。

## 运行主实验

主实验配置包括：

```text
configs/ham10000_centralized.yaml
configs/ham10000_nodp.yaml
configs/ham10000_gaussian.yaml
configs/ham10000_laplace.yaml
configs/ham10000_hybrid.yaml
configs/ham10000_hybrid_adaptive.yaml
```

批量运行：

```bash
python scripts/run_formal_experiments.py --group main --skip-existing
```

单独运行某个实验：

```bash
python scripts/run_experiment.py --config configs/ham10000_hybrid.yaml
```

每个实验会生成：

```text
outputs/<experiment_name>/
  summary.json
  history.csv
  history.json
  curves.png
  best_model.pt
  val_predictions_best.npz
  test_predictions_best.npz
  final_model.pt
```

## 运行补充实验

高精度补充实验：

```bash
python scripts/run_formal_experiments.py --group highacc --skip-existing
```

多种子稳定性实验：

```bash
python scripts/run_multi_seed_experiments.py \
  --config configs/ham10000_literature_target.yaml \
  --seeds 2026 2027 2028 \
  --rounds 60
```

多种子脚本会生成：

```text
configs/generated/multi_seed/ham10000_literature_target_seed2026.yaml
configs/generated/multi_seed/ham10000_literature_target_seed2027.yaml
configs/generated/multi_seed/ham10000_literature_target_seed2028.yaml
outputs/ham10000_literature_target_seed2026/
outputs/ham10000_literature_target_seed2027/
outputs/ham10000_literature_target_seed2028/
outputs/multi_seed_summary.json
outputs/multi_seed_summary.csv
```

如果多种子训练已经完成，只重新汇总：

```bash
python scripts/run_multi_seed_experiments.py \
  --config configs/ham10000_literature_target.yaml \
  --seeds 2026 2027 2028 \
  --collect-only
```

## 冲刺 90% Accuracy 上限实验

`accuracy90` 组用于探索 HAM10000 病灶级固定划分下的高精度上限，不用于替代主实验的联邦/隐私机制对比。该组默认关闭隐私噪声和多客户端 Non-IID，重点验证更强骨干网络、较高分辨率、TTA、阈值搜索和预测概率集成是否能把完整测试集 Accuracy 推到 90% 以上。配置中的 `rounds` 是最大轮数，实际训练会根据验证集 Accuracy 进行 early stopping。

运行 ConvNeXt-Small 高精度多种子实验：

```bash
python scripts/run_multi_seed_experiments.py \
  --config configs/ham10000_accuracy90.yaml \
  --seeds 2026 2027 2028 2029 2030
```

运行一个不同骨干网络作为集成补充：

```bash
python scripts/run_experiment.py --config configs/ham10000_accuracy90_efficientnet_b4.yaml
```

训练完成后，对验证集最佳轮次的预测概率做加权集成，并只用验证集选择阈值。最终采用验证集 `Sensitivity >= 0.70` 作为约束，以 `Accuracy` 作为阈值选择指标：

```bash
python scripts/evaluate_prediction_ensemble.py \
  --experiments \
  ham10000_accuracy90_seed2026 \
  ham10000_accuracy90_seed2028 \
  ham10000_accuracy90_seed2030 \
  ham10000_accuracy90_efficientnet_b4 \
  ham10000_accuracy90_seed2027 \
  --weights 0.25 0.10 0.30 0.30 0.05 \
  --output-name ham10000_accuracy90_weighted_5models_sens70 \
  --threshold-metric accuracy \
  --min-sensitivity 0.70
```

集成输出：

```text
outputs/ham10000_accuracy90_weighted_5models_sens70/
  summary.json
  summary.csv
  history.csv
  history.json
  val_predictions_ensemble.npz
  test_predictions_ensemble.npz
```

当前固定划分下，该加权五模型集成结果为：

```text
Accuracy:     0.9057
AUC:          0.9417
F1:           0.8502
Sensitivity:  0.7363
Specificity:  0.9485
Threshold:    0.7400
```

判断是否真正超过 90%，应看完整测试集 `test_accuracy_at_best_val > 0.9000`，同时确认 `test_sensitivity_at_best_val` 不低于主实验水平，避免只靠偏向 benign 类别获得虚高 Accuracy。等权 6 模型集成不一定优于加权集成，本文最终高精度上限结果以 `ham10000_accuracy90_weighted_5models_sens70` 为准。

## 运行消融实验

已有实验组：

```bash
python scripts/run_formal_experiments.py --group noise_ablation --skip-existing
python scripts/run_formal_experiments.py --group noniid_ablation --skip-existing
```

生成并运行更多消融配置：

```bash
python scripts/run_ablation_experiments.py --generate-only
python scripts/run_ablation_experiments.py --skip-existing --collect
```

消融实验覆盖客户端数量、裁剪阈值、Non-IID 程度和噪声强度。

## 成员推断攻击评估

攻击评估依赖对应实验的 `final_model.pt`。先确保主实验模型存在，然后运行：

```bash
python scripts/run_attack_experiment.py \
  --experiments ham10000_nodp ham10000_gaussian ham10000_hybrid ham10000_hybrid_adaptive
```

输出文件：

```text
outputs/attack_summary.json
outputs/attack_summary.csv
```

攻击准确率越接近 50%，表示成员推断攻击越接近随机猜测。

## 汇总结果并打开看板

训练完成后刷新实验汇总：

```bash
python scripts/collect_experiment_results.py --target-accuracy 0.80
python scripts/run_multi_seed_experiments.py \
  --config configs/ham10000_literature_target.yaml \
  --seeds 2026 2027 2028 \
  --collect-only
python scripts/build_dashboard_data.py --experiment-name ham10000_literature_target
```

默认刷新看板时不会加载模型生成样例预测，只会写入良性/恶性各 2 张的平衡真值样例，避免把标签伪装成预测。若答辩前需要样例卡显示真实模型预测，可显式运行：

```bash
python scripts/build_dashboard_data.py --experiment-name ham10000_literature_target --sample-predictions --sample-mode single
```

启动答辩演示服务：

```bash
python scripts/serve_demo.py --preload-models
```

启动后会自动选择空闲端口并打开浏览器，终端会打印实际访问地址，例如：

```text
答辩演示地址: http://127.0.0.1:8001/
```

实际端口以终端输出为准。如果不想自动打开浏览器，可加 `--no-browser`：

```bash
python scripts/serve_demo.py --preload-models --no-browser
```

如果现场机器启动五模型集成较慢，可临时切到单模型备用：

```bash
python scripts/serve_demo.py --mode single --preload-models
```

现场识别模块默认读取 `outputs/` 里的现成权重文件；如果这些文件不存在，先准备对应实验输出再启动服务。

## 从零复现推荐顺序

```bash
# 1. 安装依赖后准备数据
python scripts/prepare_ham10000.py --src data/raw_ham10000 --dst data/ham10000_binary --group-field lesion_id --seed 2026
python scripts/audit_dataset.py --root data/ham10000_binary

# 2. 运行主实验
python scripts/run_formal_experiments.py --group main --skip-existing

# 3. 运行补充实验和多种子实验
python scripts/run_formal_experiments.py --group highacc --skip-existing
python scripts/run_multi_seed_experiments.py --config configs/ham10000_literature_target.yaml --seeds 2026 2027 2028 --rounds 60

# 4. 可选运行 90% Accuracy 上限实验
python scripts/run_multi_seed_experiments.py --config configs/ham10000_accuracy90.yaml --seeds 2026 2027 2028 2029 2030
python scripts/run_experiment.py --config configs/ham10000_accuracy90_efficientnet_b4.yaml
python scripts/evaluate_prediction_ensemble.py --experiments ham10000_accuracy90_seed2026 ham10000_accuracy90_seed2028 ham10000_accuracy90_seed2030 ham10000_accuracy90_efficientnet_b4 ham10000_accuracy90_seed2027 --weights 0.25 0.10 0.30 0.30 0.05 --output-name ham10000_accuracy90_weighted_5models_sens70 --threshold-metric accuracy --min-sensitivity 0.70

# 5. 运行消融和攻击评估
python scripts/run_ablation_experiments.py --skip-existing --collect
python scripts/run_attack_experiment.py --experiments ham10000_nodp ham10000_gaussian ham10000_hybrid ham10000_hybrid_adaptive

# 6. 刷新汇总并启动答辩演示
python scripts/collect_experiment_results.py --target-accuracy 0.80
python scripts/run_multi_seed_experiments.py --config configs/ham10000_literature_target.yaml --seeds 2026 2027 2028 --collect-only
python scripts/build_dashboard_data.py --experiment-name ham10000_literature_target
# 可选：需要样例卡显示真实模型预测时，改用下面这一行刷新看板数据
# python scripts/build_dashboard_data.py --experiment-name ham10000_literature_target --sample-predictions --sample-mode single
python scripts/serve_demo.py --preload-models
```

## 结果口径

本项目默认使用验证集选轮报告测试集指标，选轮规则记录在 `summary.json` 的 `test_selection_rule` 字段中。不要用测试集最高轮次作为主结果。

主要结果文件：

```text
outputs/experiment_summary.csv
outputs/experiment_summary.json
outputs/multi_seed_summary.csv
outputs/multi_seed_summary.json
outputs/attack_summary.csv
outputs/attack_summary.json
```

多种子结果以 `mean ± std` 形式展示，适合说明训练稳定性。

## 注意事项

- `prepare_ham10000.py` 会重建目标数据目录，运行前请确认 `--dst` 指向的是可覆盖的数据集输出目录。
- 当前隐私预算是客户端更新级扰动下的近似分析。Gaussian 机制显示近似 epsilon，Hybrid 显示 Gaussian 部分近似 epsilon，Laplace 未实现严格 epsilon 会计。
- `final_model.pt` 和 HAM10000 图像体积较大，建议通过网盘、Release 或实验环境单独管理，不建议直接提交到 GitHub。
