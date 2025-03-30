# import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F

class OneDimVAE(nn.Module):
    def __init__(self, latent_dim, input_length = 1929928, kernel_size=7, divide_slice_length=4096,kld_weight=0.005):
        super(OneDimVAE, self).__init__()
        d_model = [8,16, 32, 64, 128, 256, 256, 128, 64, 32, 16,8]
        self.d_model = d_model
        self.d_latent = latent_dim
        self.kld_weight = kld_weight
        self.divide_slice_length = divide_slice_length
        self.initial_input_length = input_length
        # confirm self.last_length
        input_length = (input_length // divide_slice_length + 1) * divide_slice_length \
                if input_length % divide_slice_length != 0 else input_length
        assert input_length % int(2 ** len(d_model)) == 0, \
                f"Please set divide_slice_length to {int(2 ** len(d_model))}."

        self.adjusted_input_length = input_length
        self.last_length = input_length // int(2 ** len(d_model))

        # Build Encoder
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, 2, kernel_size//2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Linear(self.last_length * d_model[-1], latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # Build Decoder
        modules = []
        self.to_decode = nn.Linear(latent_dim, self.last_length * d_model[-1])
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i+1], kernel_size, 2, kernel_size//2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.ELU(),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, 2, kernel_size//2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, 1, kernel_size//2),
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
    def encode(self, input, **kwargs):
        # print(input.shape)
        # assert input.shape == [batch_size, num_parameters]
        # input = input[:, None, :]
        # Check input dimensions
        if input.dim() == 2:  # [batch_size, sequence_length]
            input = input[:, None, :]  # Add channel dimension
        elif input.dim() == 3:  # [batch_size, 1, sequence_length]
            pass  # Input shape is already correct

            # 填充序列
        input = self.pad_sequence(input)  # [B, 1, adjusted_input_length]

        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        # result = self.to_decode(z)
        # result = result.view(-1, self.d_model[-1], self.last_length)
        # result = self.decoder(result)
        # result = self.final_layer(result)
        # assert result.shape[1] == 1, f"{result.shape}"
        result = self.to_decode(z)
        # print(f"After to_decode: {result.shape}")
        result = result.view(-1, self.d_model[-1], self.last_length)
        # print(f"After reshape: {result.shape}")
        result = self.decoder(result)
        # print(f"After decoder: {result.shape}")
        result = self.final_layer(result)
        # print(f"After final_layer: {result.shape}")
        return result[:, 0, :]

    def reparameterize(self, mu, log_var, **kwargs):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_decode(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var

    def sample(self, batch=1):
        z = torch.randn((batch, self.d_latent), device=self.device, dtype=torch.float32)
        recons = self.decode(z)
        return recons

    def forward(self, x, **kwargs):
        recons, input, mu, log_var = self.encode_decode(input=x, **kwargs)
        # recons = recons.view(input.shape)  # 调整 recons 的形状
        # recons = recons.view(x.size(0), self.adjusted_input_length)  # 调整 recons 的形状

        # recons_loss = F.mse_loss(recons, input)

        padded_x = self.pad_sequence(x)
        recons_loss = F.mse_loss(recons, padded_x, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss +self.kld_weight * kld_loss

        return loss, recons_loss, kld_loss

    @property
    def device(self):
        return next(self.parameters()).device