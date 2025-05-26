# In-Context Meta LoRA Generation (IJCAI2025)

[English](README.en.md) | [简体中文](README.md)  

![Pipeline](images/pipeline.jpg)

## 📌 数据准备

### 📂 数据集准备
请将数据集下载至 `data` 文件夹。Florence2 训练格式如下：

```json
{
  "id": "image_0001",
  "image_path": "/path/to/images/image_0001.jpg",
  "height": 480,
  "width": 640,
  "suffix": "dog<loc_120><loc_85><loc_340><loc_250> person<loc_50><loc_30><loc_150><loc_200> car<loc_400><loc_150><loc_600><loc_300>"
}
```

---

## 🔧 训练 LoRA 参数
使用 `train_lora` 文件夹下的 `train_lora_arg.py` 脚本对大模型进行 LoRA 微调，以生成不同 rank（1-8）的 LoRA 参数。

---

## 🔄 处理 LoRA 参数
### 🔹 展平并归一化 LoRA 参数
1. 运行 `utils/reformat_lora_param.py` 脚本，将原始 LoRA 参数标注 `epoch` 并移动到 `param` 文件夹。
2. 运行 `utils/normalizeLoraWeight_small.py` 脚本，对 LoRA 参数进行展平和归一化。
3. 观察展平后 LoRA 参数的维度，这将成为 CVAE 的 `input_dim`。

```bash
#!/bin/bash

SOURCE_PATH=../train_lora/model_checkpoints/xxxx # 例如 dog-r=8
TARGET_PATH=../data/param_data/xxx  # 例如 dog-r=8

python3 reformat_lora_param.py --source_path "$SOURCE_PATH" --target_path "$TARGET_PATH"
python3 normalizeLoraWeight_small.py --dataset_path "$TARGET_PATH"
```

---

## 🏗️ 训练参数生成器
### 🔹 获取 Task Vector
使用 `ICL/last_time_step_icl.py` 脚本：
- 修改数据集路径。
- 指定 LoRA 微调模型。
- 设置 `output_dir`，以获取最终的 `hidden_state` (`decoder_avg_hidden_state`)。
- 观察 `task_vector` 形状，这将成为 CVAE 的 `condition_dim`。

### 🔹 训练 CVAE
使用 `TrainScript_CVAE.py` 训练 CVAE 模型。

```bash
python3 TrainScript_CVAE.py \
  --train_data_dir ../data/xxxx/normalized_data \
  --condition_dim xxx \
  --input_dim xxxxx \
  --task_vector_path xxxxx \
  --checkpoint_dir xxx
```

---

## 🔄 LoRA 重建
使用 `utils/ReconstructLora_cvae.py` 进行 LoRA 采样与重建。

```bash
python3 ReconstructLora_cvae.py \
  --train_data_dir ../data/xxxx/normalized_data \
  --condition_dim xxx \
  --input_dim xxxxx \
  --task_vector_path xxxxx \
  --cvae_checkpoint_path xxx \
  --datasetname "dog-r=8" \
  --normalized_lora_path xxx \
  --rank 8 # 可选：1, 2, 4, 8
```

---

## ✅ 测试
使用 `test.py` 进行测试。

```bash
python3 test.py \
  --download_location xxx \
  --datasetname xxx \
  --generated_lora xxx \
  --rank xxx
```


