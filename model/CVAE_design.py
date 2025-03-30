import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, latent_dim, condition_dim = 512, input_length=1929928, kernel_size=7, divide_slice_length=4096, kld_weight=0.005):
        super(CVAE, self).__init__()
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # 原始 VAE 结构
        d_model = [8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8]
        self.d_model = d_model
        self.kld_weight = kld_weight
        self.divide_slice_length = divide_slice_length
        self.initial_input_length = input_length

        # 确定最后一层的长度
        input_length = (input_length // divide_slice_length + 1) * divide_slice_length \
            if input_length % divide_slice_length != 0 else input_length
        assert input_length % int(2 ** len(d_model)) == 0, \
            f"Please set divide_slice_length to {int(2 ** len(d_model))}."

        self.adjusted_input_length = input_length
        self.last_length = input_length // int(2 ** len(d_model))

        #编码器部分
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, 2, kernel_size // 2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Linear(self.last_length * d_model[-1], latent_dim)
        self.fc_mu = nn.Linear(latent_dim + condition_dim, latent_dim)  # 拼接 c
        self.fc_var = nn.Linear(latent_dim + condition_dim, latent_dim)

        #解码器部分
        modules = []
        self.to_decode = nn.Linear(latent_dim + condition_dim, self.last_length * d_model[-1])  # 拼接 c
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i + 1], kernel_size, 2, kernel_size // 2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.ELU(),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, 2, kernel_size // 2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, 1, kernel_size // 2),
        )

    def pad_sequence(self, input_seq):
        """
        在序列末尾添加零以调整长度到 self.adjusted_input_length。
        """
        batch_size, channels, seq_length = input_seq.size()
        if seq_length < self.adjusted_input_length:
            padding_size = self.adjusted_input_length - seq_length
            # 在最后一个维度上填充
            input_seq = F.pad(input_seq, (0, padding_size), "constant", 0)
        elif seq_length > self.adjusted_input_length:
            # 截断多余的部分
            input_seq = input_seq[:, :, :self.adjusted_input_length]
        return input_seq

    def encode(self, input, c):
        """
        编码器：接收输入 x 和类别向量 c，返回潜在变量的均值和方差。
        """
        if input.dim() == 2:  # [batch_size, sequence_length]
            input = input[:, None, :]  # Add channel dimension

        input = self.pad_sequence(input)  # [B, 1, adjusted_input_length]
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
    
        # 将类别向量 c 拼接到结果中
        result = torch.cat([result, c], dim=-1)  # [batch_size, latent_dim + condition_dim]
                

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, c):
        """
        解码器：接收潜在变量 z 和类别向量 c，返回重建结果。
        """
        # 拼接 z 和 c
        z = torch.cat([z, c], dim=-1)  # [batch_size, latent_dim + condition_dim]
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[-1], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result[:, 0, :]  # [batch_size, sequence_length]

    def forward(self, x, c):
        """
        前向传播：接收输入 x 和类别向量 c，返回总损失和各部分损失。
        """
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z, c)

        padded_x = self.pad_sequence(x)
        recons_loss = F.mse_loss(recons, padded_x, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss

        return loss, recons_loss, kld_loss

    @property
    def device(self):
        return next(self.parameters()).device
