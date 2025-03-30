# @title Imports

import argparse
import csv
import os
import re
import torch

import numpy as np
import supervision as sv
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor
)
from roboflow import Roboflow
from peft import PeftModel
from peft import PeftModel, PeftConfig
from data.florence_detection_dataset import DetectionDataset


# 解析命令行参数
parser = argparse.ArgumentParser(description="Run Florence-2 object detection with LoRA.")
parser.add_argument("--download_location", required=True, help="Path to dataset download location.")
parser.add_argument("--lora_rank", required=True, type=int, choices=[1, 2, 4, 8], help="LoRA rank.")
parser.add_argument("--datasetname", required=True, help="Dataset name (e.g., chair, bicycle, bus).")
args = parser.parse_args()

# 解析参数
download_location = args.download_location
lora_rank = str(args.lora_rank)
datasetname = args.datasetname


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_hook_path = "./hidden_state/"


if not os.path.exists(save_hook_path):
    os.makedirs(save_hook_path)
    print(f"目录 {save_hook_path} 已创建")
else:
    print(f"目录 {save_hook_path} 已存在")


def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

#TODO: 修改数据集位置 e.g. bicycle, bus
train_dataset = DetectionDataset(
    jsonl_file_path = f"{download_location}/train/annotations_"+ datasetname + "jsonl",
    image_directory_path = f"{download_location}/train/"
)
val_dataset = DetectionDataset(
    jsonl_file_path = f"{download_location}/valid/annotations_" + datasetname + ".jsonl",
    image_directory_path = f"{download_location}/valid/"
)


#TODO: 修改为对应的生成模型的位置
# peft_model_id = "/home/ma-user/work/ymxwork/ymx/Neural-Network-Parameter-Diffusion-main/florence2-lora/sofa-voc"
peft_model_id = "../train_lora/model_checkpoints/" + datasetname + "-r=" + lora_rank + "/epoch_49"
config = PeftConfig.from_pretrained(peft_model_id)
config.base_model_name_or_path = "models/florence2"
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True).to(DEVICE)
model = PeftModel.from_pretrained(model, peft_model_id).to(DEVICE)
processor = AutoProcessor.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, revision=REVISION)

PATTERN = r'([a-zA-Z0-9_ ]+)<loc_\d+>'

"""
    EXTRACT CLASSES FORM DATASET
"""
def extract_classes(dataset: DetectionDataset):
    class_set = set()
    for i in range(len(dataset.dataset)):
        image, data = dataset.dataset[i]
        suffix = data["suffix"]
        classes = re.findall(PATTERN, suffix)
        class_set.update(classes)
    return sorted(class_set)

CLASSES = extract_classes(train_dataset)

targets = []
predictions = []

REVISION = 'refs/pr/6'


###################################注册钩子函数##########################################3
import torch.nn.functional as F

# Hook Functions
decoder_outputs = []


def replace_hidden_state_hook_decoder(module, input, output):
    last_time_step = output[0][:, -1, :].detach().cpu()
    decoder_outputs.append(last_time_step)


############################注册钩子函数######################################


last_layer_decoder = model.language_model.get_decoder().layers[-1] 
hook_handle_decoder = last_layer_decoder.register_forward_hook(replace_hidden_state_hook_decoder)

for i in range(30):
    image, data = train_dataset.dataset[i]
    prefix = data['prefix']
    suffix = data['suffix']

    inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(generated_text)

    prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
    prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
    prediction = prediction[np.isin(prediction['class_name'], CLASSES)]
    prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
    prediction.confidence = np.ones(len(prediction))

    target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
    target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
    target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

    targets.append(target)
    predictions.append(prediction)

####回收钩子######
hook_handle_decoder.remove()

mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)

print(f"map50_95: {mean_average_precision.map50_95:.2f}")
print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")

# 保存结果到 CSV 文件
result_dir = f"./results/lora_inference_ICL30/" + datasetname
os.makedirs(result_dir, exist_ok=True)

csv_file_path = os.path.join(result_dir, "results.csv")
header = ['Metric', 'Value']
rows = [
    ['map50_95', f"{mean_average_precision.map50_95:.2f}"],
    ['map50', f"{mean_average_precision.map50:.2f}"],
    ['map75', f"{mean_average_precision.map75:.2f}"]
]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)


# 计算 encoder 和 decoder 输出的平均值
decoder_avg = torch.mean(torch.stack(decoder_outputs), dim=0)
decoder_avg = torch.mean(decoder_avg, dim=0)

print(f"Decoder Average Shape: {decoder_avg.shape}")

# 保存为 .pth 文件
output_dir = "./hidden_state/" + datasetname + "_vector_avg/lora_r="+ lora_rank + "/"
os.makedirs(output_dir, exist_ok=True)
decoder_path = os.path.join(output_dir, "hidden_state.pth")

torch.save(decoder_avg, decoder_path)
print(f"Decoder average hidden state saved to {decoder_path}")