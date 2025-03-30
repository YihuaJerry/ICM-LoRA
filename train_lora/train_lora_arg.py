import argparse
from ast import parse
import os
import torch

from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from PIL import Image
from roboflow import Roboflow
from data.florence_detection_dataset import DetectionDataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def train_with_dataset(dataset_name: str, lora_r: int, download_location: str):
    CHECKPOINT = "/models/florence2"
    REVISION = 'refs/pr/6'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION).to(DEVICE)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True, revision=REVISION)


    BATCH_SIZE = 6
    NUM_WORKERS = 0

    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
        return inputs, answers
    
    # 从dataset_name中提取基础名称（不包含r=部分）
    base_name = dataset_name.split('-r=')[0]
    
    train_dataset = DetectionDataset(
        jsonl_file_path=f"{download_location}/train/annotations_{base_name}.jsonl",
        image_directory_path=f"{download_location}/train/"
    )
    val_dataset = DetectionDataset(
        jsonl_file_path=f"{download_location}/valid/annotations_{base_name}.jsonl",
        image_directory_path=f"{download_location}/valid/"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    config = LoraConfig(
        r=lora_r,  # 使用传入的 r 参数
        lora_alpha=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
        revision=REVISION
    )

    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()

    torch.cuda.empty_cache()

    def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
        optimizer = AdamW(model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    labels = processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False
                    ).input_ids.to(DEVICE)

                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Average Validation Loss: {avg_val_loss}")
            if epoch >= 100:
                output_dir = f"./model_checkpoints/{dataset_name}/epoch_{epoch -100 + 1}"
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                processor.save_pretrained(output_dir)

    EPOCHS = 200
    LR = 5e-6

    train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR)

    output_dir = f"./florence2-lora/{dataset_name}-r={lora_r}"
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("========= Save Model Done ===========")

def main():
    parser = argparse.ArgumentParser(description='Train Lora for a category, including r=1,r=2,r=4,r=8')
    parser.add_argument("--dataset_name", required=True, help='Dataset Name (base name without r parameter)')
    parser.add_argument("--download_location", required=True, help='Directory where dataset is stored')
    args = parser.parse_args()
    
    # 定义不同的 LoRA r 参数进行训练
    lora_r_values = [1, 2, 4, 8]
    
    # 使用命令行参数的数据集名称作为基础名称
    dataset_base_name = args.dataset_name
    download_location = args.download_location

    for r in lora_r_values:
        dataset_name = f"{dataset_base_name}-r={r}"
        train_with_dataset(dataset_name, lora_r=r, download_location=download_location)

if __name__ == "__main__":
    main()