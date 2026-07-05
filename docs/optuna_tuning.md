# Optuna 调参脚本

新增的 `scripts/tune_optuna.py` 会复用现有 Qlib workflow YAML：

- `qlib_init` 用来初始化 Qlib 数据源和实验配置。
- `task.dataset` 只构建一次，所有 trial 复用同一份数据集。
- `task.model` 作为基础模型配置，每个 trial 只覆盖 Optuna 采样到的参数。
- objective 默认最大化 `QlibGCN.fit(..., evals_result=...)` 里的验证集分数；当前 YAML 使用 `metric: ic`，因此就是最大化 valid IC。

## 安装依赖

```powershell
pip install -e .
pip install optuna PyYAML
```

或直接用更新后的 `requirements.txt` 安装。

## 推荐先小规模试跑

完整 GCN 训练很慢，建议先缩短每个 trial 的训练轮数：

```powershell
python scripts/tune_optuna.py `
  --config examples/workflow_config_gcn_Alpha360.yaml `
  --n-trials 20 `
  --trial-epochs 40 `
  --trial-early-stop 8 `
  --storage sqlite:///optuna_studies/qlib_gcn_optuna.db `
  --output-json optuna_best_params.json `
  --save-best-config examples/workflow_config_gcn_Alpha360_optuna_best.yaml
```

如果只想在内存里试跑，不保留 study：

```powershell
python scripts/tune_optuna.py --n-trials 3 --trial-epochs 5 --storage none
```

## 默认搜索空间

- `lr`: `1e-5` 到 `3e-3`，log scale
- `weight_decay`: `1e-8` 到 `1e-3`，log scale
- `dropout`: `0.0` 到 `0.5`
- `hidden_size`: `32 / 64 / 128`
- `gcn_hidden_size`: `32 / 64 / 128`
- `num_layers`: `1` 到 `3`
- `topk`: `5 / 10 / 20 / 30 / 50`
- `min_edge_weight`: `0.0` 到 `0.3`

默认不调 `base_model`，继续使用 YAML 里的 `GRU`。如果也想比较 `GRU` 和 `LSTM`，加：

```powershell
python scripts/tune_optuna.py --tune-base-model
```

## 输出

脚本会写出：

- `optuna_best_params.json`: 最佳 trial、最佳分数、采样参数、合并后的完整模型参数。
- `--save-best-config` 指定的 YAML: 把最佳模型参数合并进 `task.model.kwargs`，方便后续 `qrun`。

注意：生成的 best config 是普通 YAML，会保留配置值，但不会保留原文件里的注释和 anchor 写法。

调参时建议只看 valid IC；最终结果仍然应该用最佳参数在 test 段单独跑一次完整 workflow 和回测。
