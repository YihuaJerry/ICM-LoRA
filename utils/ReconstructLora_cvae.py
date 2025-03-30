import argparse
import re
from turtle import st
import torch
import os
from safetensors.torch import load_file
import sys
from safetensors.torch import save_file
# 导入 CVAE 模型
from model.CVAE_design import CVAE


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training Script for VAE")

    #condition dim, the length of task vector(get from last_time)
    parser.add_argument('--condition_dim',type=int, nargs='+',default=768,help="condition dim, the length of task vector(get from last_time)")

    #input_dim, the dim of lora param
    parser.add_argument('--input_dim', type=int, nargs='+',default=1929928,help="input_dim, the dim of lora param")

    #task_vector path
    parser.add_argument('--task_vetcor_path',type=str,required=True,default="../ICL/hidden_state/e.g dog-r=8/hidden_state.pth")

    parser.add_argument('--cvae_checkpoint_path',type=str, required=True,default="./checkpoints/lora_cvae_checkpoints/checkpoint_End")

    parser.add_argument('--datasetname',required=True,type=str,help="e.g dog")

    parser.add_argument('--rank',required=True,type=str,help="1,2,4,8")

    parser.add_argument('--normalized_lora_path',required=False, type = str, default="../data/param_data/"+ args.datasetname + "/normalized_data/normalized_adapter_model_99.pth")

    args = parser.parse_args()
    print(args)

    # ==============================
    # 1. 设置模型参数
    # ==============================
    latent_dim = 256            # 与训练时相同
    input_length = args.input_dim     # 与训练数据长度相同
    condition_dim = args.condition_dim         # 条件向量的维度
    kld_weight = 0.005          # KL 散度损失权重

    # ==============================
    # 2. 创建模型实例并加载权重
    # ==============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CVAE(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        input_length=input_length,
        kld_weight=kld_weight
    ).to(device)

    # 加载模型权重
    checkpoint_path = args.cvae_checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    model.load_state_dict(load_file(checkpoint_path))
    print("Model weights loaded successfully.")

    model.eval()

    # ==============================
    # 3. 采样生成数据
    # ==============================
    num_samples = 3  # 生成样本的数量

    encoder_avg_hidden_state = torch.load(args.task_vetcor_path)
    print("Loaded encoder average hidden state shape:", encoder_avg_hidden_state.shape)

    category_vector = encoder_avg_hidden_state
    category_vector = category_vector.squeeze(dim=1)
    print("Shape after squeezing:", category_vector.shape)

    # 如果它仍然是二维的，您可以直接将其展平：
    category_vector = category_vector.view(-1)
    print("Shape after flattening:", category_vector.shape)

    category_vector = category_vector.unsqueeze(0)##加一个batch size
    print("Shape after unsqueeze:", category_vector.shape)

    condition = category_vector.repeat(num_samples, 1).to(model.device)

    #TODO: move this to cvae design
    with torch.no_grad():
        # 从标准正态分布中采样潜在向量 z
        z = torch.randn((num_samples, latent_dim)).to(device)
        generated_data = model.decode(z, condition).to(device)

    # ==============================
    # 4. 保存生成的数据
    # ==============================
    dataset_name = args.datasetname
    output_dir = '../generated_samples/'+ dataset_name + "-r=" + args.rank
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        sample = generated_data[i].cpu()
        sample_path = os.path.join(output_dir, f'sample_{i + 1}.pth')
        sample = sample[:input_length]  # 确保样本长度一致
        torch.save(sample, sample_path)
        print(f"Sample {i + 1} saved to {sample_path}")

    sample1_path = os.path.join(output_dir,'sample_1.pth')
    print("All samples generated and saved successfully.")

    reconstructed_lora_vector = torch.load(sample1_path)

    # 打印重建模型参数信息
    reconstructed_lora_param_info = {}

    if isinstance(reconstructed_lora_vector, dict):
        for key, value in reconstructed_lora_vector.items():
            print(key, value.shape)
            reconstructed_lora_param_info[key] = {'shape': value.shape, 'length': value.numel()}
    else:
        print(reconstructed_lora_vector)
        print("Loaded data is not a dictionary. It might be a single Tensor.")

    print(reconstructed_lora_vector[:1000])

    # TODO:从文件中加载数据字典
    data_path = args.normalized_lora_path

    data_dict = torch.load(data_path)

    # 移除数据字典中的 'data' 键
    data_keys = [k for k in data_dict.keys() if k != 'data']

    flattened_data = reconstructed_lora_vector

    # 获取数据的形状和长度
    lengths = [data_dict[k]['length'] for k in data_keys]

    # 分割展平的数据为每个参数的长度
    print(f"Input tensor shape: {flattened_data.shape}")
    total_length = sum(lengths)
    print(f"Total length from split_sizes: {total_length}")
    print(f"Flattened data size: {flattened_data.shape[0]}")

    # 如果flattened_data不是一个1D张量，则将其第一维去掉
    if len(flattened_data.shape) > 1:
        flattened_data = flattened_data.squeeze()
        print(f"Flattened data squeezed to shape: {flattened_data.shape}")
    split_data = torch.split(flattened_data, lengths)

    # 初始化
    restored_state_dict = {}

    # 循环遍历每个参数
    for i, key in enumerate(data_keys):
        # 重建参数
        data = split_data[i]
        # 从数据字典中获取均值和标准差
        mean = data_dict[key]['mean']
        std = data_dict[key]['std']
        # 逆归一化
        denormalized_data = data * std + mean

        restored_state_dict[key] = denormalized_data
    #     重建为原始的形状
        restored_state_dict[key] = denormalized_data.reshape(data_dict[key]['shape'])

    save_path = '/output_reconstructed/'+ dataset_name
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save(restored_state_dict, os.path.join(save_path, 'restored_state_dict.pth'))
    save_file(restored_state_dict, os.path.join(save_path, 'adapter_model.safetensors'))
    print(f"Restored parameters have been saved to 'restored_state_dict.pth'")



if __name__ == '__main__':
    main()
