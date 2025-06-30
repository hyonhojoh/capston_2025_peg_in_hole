import torch

import torch.nn as nn

import torch.optim as optim

import os

import torch.optim.lr_scheduler as lr_scheduler

from datetime import datetime

from dataclasses import dataclass, asdict

from torch.utils.data import DataLoader, random_split # random_split 추가


# --- WandB Import ---

import wandb


from model import Autoencoder

from data_loader import get_image_dataloader

from utils import prepare_theia_transformer_model, get_theia_features_shape



@dataclass(frozen=True)

class TrainingConfig:

    epochs: int = 30

    batch_size: int = 64

    learning_rate: float = 5e-4

    latent_dim: int = 32

    weight_decay: float = 1e-5

    model_name: str = "theia-tiny-patch16-224-cddsv"

    architecture: str = "SimpleAutoencoder_BN_LeakyReLU"

    validation_split: float = 0.1 # 검증 세트 비율 (e.g., 10%)



############ Configuration ############

# (이전과 동일)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224

NUM_CHANNELS = 3

camera_index = str(1)

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', camera_index, 'rgb')

########################################



def train():

    cfg = TrainingConfig()
    run = wandb.init(project="peg-in-hole-autoencoder", config=asdict(cfg))

    checkpoint_save_dir = os.path.join(wandb.run.dir, 'checkpoints')

    os.makedirs(checkpoint_save_dir, exist_ok=True)

    

    print(f"Using device: {DEVICE}")

    print(f"WandB Run Dir: {wandb.run.dir}")


    # Theia 모델 로딩

    print(f"Loading pretrained model: {cfg.model_name}")

    pretrained_model_config = prepare_theia_transformer_model(model_name=cfg.model_name, model_device=DEVICE)

    pretrained_model = pretrained_model_config['model']()

    pretrained_inference_fn = pretrained_model_config['inference']


    FEATURE_DIM = get_theia_features_shape(pretrained_model, (1, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), DEVICE)

    print(f"Feature Dimension from [CLS] token: {FEATURE_DIM}")


    # --- [수정] 데이터셋 로드 및 분할 ---

    # get_image_dataloader는 전체 데이터셋을 포함한 DataLoader를 반환합니다.

    # 우리는 그 안의 dataset 객체만 필요합니다.

    full_dataloader = get_image_dataloader(

        data_root=DATA_ROOT,

        image_height=IMAGE_HEIGHT,

        image_width=IMAGE_WIDTH,

        batch_size=cfg.batch_size,

        normalize=False,

        num_workers=os.cpu_count() // 2 or 1,

        shuffle=True, # 섞어서 분할하는 것이 좋습니다.

    )

    full_dataset = full_dataloader.dataset


    # 데이터셋 분할

    val_size = int(cfg.validation_split * len(full_dataset))

    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    print(f"Full dataset size: {len(full_dataset)}")

    print(f"Training set size: {len(train_dataset)}")

    print(f"Validation set size: {len(val_dataset)}")


    # 각 데이터셋에 대한 DataLoader 생성

    train_loader = DataLoader(

        train_dataset, batch_size=cfg.batch_size, shuffle=True, 

        num_workers=os.cpu_count() // 2 or 1, pin_memory=True

    )

    val_loader = DataLoader(

        val_dataset, batch_size=cfg.batch_size, shuffle=False, # 검증 시에는 섞을 필요 없음

        num_workers=os.cpu_count() // 2 or 1, pin_memory=True

    )

    # --- 데이터셋 분할 종료 ---


    model = Autoencoder(FEATURE_DIM, cfg.latent_dim).to(DEVICE)

    print(f"Autoencoder initialized with Input: {FEATURE_DIM}, Latent: {cfg.latent_dim}")


    wandb.watch(model, log="all", log_freq=100)


    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    

    print("\n--- Starting Autoencoder Training ---")


    best_val_loss = float('inf')


    for epoch in range(cfg.epochs):

        # --- 훈련 루프 ---

        model.train()

        total_train_loss = 0

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{cfg.epochs}, Current Learning Rate: {current_lr:.6f}")


        for batch_idx, (images, _) in enumerate(train_loader):

            images = images.to(DEVICE)

            

            with torch.no_grad():

                img_features = pretrained_inference_fn(pretrained_model, images)


            output = model(img_features)

            loss = criterion(output, img_features)


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_train_loss += loss.item()


        avg_train_loss = total_train_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{cfg.epochs}] finished. Average Training Loss: {avg_train_loss:.6f}")


        # --- [추가] 검증 루프 ---

        model.eval() # 모델을 평가 모드로 전환

        total_val_loss = 0

        with torch.no_grad(): # 검증 시에는 그래디언트 계산이 필요 없음

            for images, _ in val_loader:

                images = images.to(DEVICE)

                

                # Theia 추론은 이미 no_grad 컨텍스트를 내부적으로 포함할 수 있지만,

                # 전체 루프를 감싸는 것이 더 안전합니다.

                img_features = pretrained_inference_fn(pretrained_model, images)


                output = model(img_features)

                loss = criterion(output, img_features)

                total_val_loss += loss.item()


        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.6f}")

        # --- 검증 루프 종료 ---


        # WandB에 로깅 (훈련 및 검증 손실 모두)

        wandb.log({

            "epoch": epoch + 1,

            "avg_train_loss": avg_train_loss,

            "avg_val_loss": avg_val_loss, # 검증 손실 추가

            "learning_rate": current_lr

        })


        scheduler.step()


        # [수정] 최고 모델 저장 기준을 검증 손실로 변경

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            best_model_path = os.path.join(checkpoint_save_dir, 'best_model.pth')

            torch.save({

                'epoch': epoch + 1,

                'model_state_dict': model.state_dict(),

                'loss': best_val_loss, # 최고 검증 손실 기록

                'feature_dim': FEATURE_DIM,

                'latent_dim': cfg.latent_dim

            }, best_model_path)

            print(f"*** New best model saved to {best_model_path} with validation loss: {best_val_loss:.6f} ***")

            

            wandb.summary["best_val_loss"] = best_val_loss

            wandb.summary["best_epoch"] = epoch + 1


    # 학습 종료 후 모델 아티팩트로 저장

    artifact = wandb.Artifact(name=f"autoencoder-{run.id}", type="model")

    artifact.add_file(best_model_path)

    run.log_artifact(artifact)


    print("--- Autoencoder Training Finished ---")

    

    run.finish()


if __name__ == "__main__":

    train()