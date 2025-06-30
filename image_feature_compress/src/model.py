import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # --- Encoder: 입력 차원을 점진적으로 줄여나가는 구조 ---
        # 192 -> 128 -> 64 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # 배치 정규화 추가
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU 사용
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, latent_dim) # 최종 잠재 공간으로 압축
        )

        # --- Decoder: 잠재 공간에서 원본 차원으로 점진적으로 복원 ---
        # latent_dim -> 64 -> 128 -> 192
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # RL 환경에서 추론 시에는 no_grad()와 함께 사용됩니다.
        return self.encoder(x)