import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from dataclasses import dataclass, asdict
import time

# --- WandB Import ---
import wandb

# --- 기존 코드에서 필요한 모듈 임포트 ---
# (model.py, data_loader.py, utils.py가 같은 디렉토리에 있거나 PYTHONPATH에 설정되어 있어야 함)
try:
    from model import Autoencoder
    from data_loader import get_image_dataloader # 데이터 로딩 및 분할에 사용
    from utils import prepare_theia_transformer_model, get_theia_features_shape
except ImportError:
    print("Make sure model.py, data_loader.py, and utils.py are in the same directory or PYTHONPATH.")
    exit()

# --- Configuration ---
@dataclass(frozen=True)
class EvaluationConfig:
    # --- WandB Artifact Configuration ---
    # !!! 사용자가 직접 채워야 하는 부분 !!!
    # 예시: wandb_project = "peg-in-hole-autoencoder"
    # 예시: wandb_run_id = "run_id_of_your_training_run" # 예: "34n5k2lp"
    # best_model.pth를 포함하는 아티팩트의 이름. train.py에서 "autoencoder-{run.id}"로 저장됨.
    # 아티팩트 이름은 wandb_run_id를 사용하여 구성할 수 있음.
    # 또는 명시적으로 artifact_name = "autoencoder-xxxxxxxx:latest" (또는 특정 버전)
    wandb_project: str = "autoencoder-evaluate" # <<--- YOUR WANDB PROJECT NAME
    wandb_run_id: str = time.strftime("%Y%m%d-%H%M%S")

    # --- Data and Model Configuration (학습 시와 동일하게 설정) ---
    batch_size: int = 64
    model_name: str = "theia-tiny-patch16-224-cddsv" # 학습 시 사용한 Theia 모델
    image_height: int = 224
    image_width: int = 224
    num_channels: int = 3
    camera_index: str = "1" # 학습 데이터와 동일한 카메라 인덱스
    validation_split: float = 0.1 # 학습 시 사용한 검증 세트 비율과 동일하게

    # --- Evaluation Specific Configuration ---
    num_samples_for_visualization: int = 1000 # t-SNE/UMAP 시각화에 사용할 샘플 수
    output_dir: str = "evaluation_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_wandb_artifact(cfg: EvaluationConfig) -> tuple[Autoencoder, int, int]:
    """WandB 아티팩트에서 모델을 로드합니다."""
    run = wandb.init(project=cfg.wandb_project, job_type="evaluation")
    
    model_path = "/home/hyunho_RCI/ai_ws/src/image_feature_compress/wandb/factory_camera_1/files/checkpoints/best_model.pth"
    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    feature_dim = checkpoint['feature_dim']
    latent_dim = checkpoint['latent_dim']
    
    model = Autoencoder(feature_dim, latent_dim).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded. Feature_dim: {feature_dim}, Latent_dim: {latent_dim}")
    run.finish() # 평가용 wandb run 종료
    return model, feature_dim, latent_dim

def evaluate_reconstruction(
    autoencoder: Autoencoder, 
    pretrained_model: nn.Module, 
    pretrained_inference_fn, 
    dataloader: DataLoader, 
    device: torch.device
) -> tuple[float, float]:
    """Autoencoder의 재구성 성능(MSE, 코사인 유사도)을 평가합니다."""
    autoencoder.eval()
    total_mse = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    criterion_mse = nn.MSELoss(reduction='sum') # 배치 내 총합

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            # 1. Pretrained ViT로 특징 추출
            original_features = pretrained_inference_fn(pretrained_model, images) # (batch_size, feature_dim)
            
            # 2. Autoencoder로 특징 재구성
            reconstructed_features = autoencoder(original_features)
            
            # 3. MSE 계산
            # MSELoss는 (N, *) 형태의 입력을 기대. 현재 (batch_size, feature_dim)이므로 문제 없음.
            mse = criterion_mse(reconstructed_features, original_features)
            total_mse += mse.item()
            
            # 4. 코사인 유사도 계산
            # F.cosine_similarity는 (N, D)와 (N, D) 입력에 대해 (N,) 텐서를 반환
            cosine_sim = F.cosine_similarity(reconstructed_features, original_features, dim=1) # 각 샘플에 대한 유사도
            total_cosine_sim += cosine_sim.sum().item() # 배치 내 유사도 합산
            
            num_batches += 1
            if num_batches % 20 == 0:
                print(f"  Processed {num_batches}/{len(dataloader)} batches for reconstruction eval...")

    avg_mse = total_mse / (len(dataloader.dataset) * original_features.shape[1]) # 샘플 수 * 특징 차원 당 평균 MSE
    # 위 방식 대신, (total_mse / num_batches) / original_features.shape[0] 로 배치 평균 MSE를 쓸 수도 있음.
    # 좀 더 일반적인 방식은 (total_mse / (len(dataloader.dataset)) 로 샘플당 평균 제곱합 오차
    # 여기서는 논문에 제시할 때 어떤 정의를 쓸지 명확히 하는게 중요.
    # feature_dim으로 나누지 않은, 벡터당 평균 MSE를 사용:
    avg_mse_per_sample = total_mse / len(dataloader.dataset)


    avg_cosine_sim = total_cosine_sim / len(dataloader.dataset) # 샘플 당 평균 코사인 유사도
    
    return avg_mse_per_sample, avg_cosine_sim

def visualize_latent_space(
    autoencoder: Autoencoder, 
    pretrained_model: nn.Module, 
    pretrained_inference_fn, 
    dataloader: DataLoader, 
    device: torch.device, 
    method: str, 
    num_samples: int,
    output_dir: str,
    latent_dim: int # 추가: plot 제목용
):
    """잠재 공간을 t-SNE 또는 UMAP으로 시각화합니다."""
    autoencoder.eval()
    latent_vectors_list = []
    
    print(f"Collecting {num_samples} latent vectors for {method.upper()} visualization...")
    collected_samples = 0
    with torch.no_grad():
        for images, _ in dataloader:
            if collected_samples >= num_samples:
                break
            images = images.to(device)
            
            original_features = pretrained_inference_fn(pretrained_model, images)
            encoded_vectors = autoencoder.encode(original_features) 
            
            latent_vectors_list.append(encoded_vectors.cpu().numpy())
            collected_samples += images.size(0)

    latent_vectors = np.concatenate(latent_vectors_list, axis=0)
    if latent_vectors.shape[0] > num_samples:
        # num_samples가 latent_vectors.shape[0] 보다 클 경우를 대비해 min 사용
        actual_num_samples_to_select = min(num_samples, latent_vectors.shape[0])
        indices = np.random.choice(latent_vectors.shape[0], actual_num_samples_to_select, replace=False)
        latent_vectors = latent_vectors[indices]
    
    # perplexity는 샘플 수보다 작아야 합니다.
    perplexity_val = min(30, latent_vectors.shape[0] - 1)
    if perplexity_val <= 0 : # 샘플이 너무 적은 경우 기본값 사용 또는 에러 처리
        print(f"Warning: Too few samples ({latent_vectors.shape[0]}) for TSNE. Setting perplexity to a default small value or skipping.")
        if latent_vectors.shape[0] <=1 :
            print("Cannot run TSNE with 1 or fewer samples. Skipping TSNE.")
            return # TSNE 실행 불가
        perplexity_val = max(1, latent_vectors.shape[0] // 2 -1 ) # 임시 방편, 최소 1
        if perplexity_val == 0 and latent_vectors.shape[0] > 1: # 그래도 0이면
             perplexity_val = latent_vectors.shape[0] -1


    print(f"Applying {method.upper()} with {latent_vectors.shape[0]} samples and perplexity {perplexity_val if method == 'tsne' else 'N/A'}...")
    if method == 'tsne':
        if latent_vectors.shape[0] <= perplexity_val: # 추가적인 안전장치
             print(f"Warning: Number of samples ({latent_vectors.shape[0]}) is less than or equal to perplexity ({perplexity_val}). Adjusting perplexity.")
             perplexity_val = max(1, latent_vectors.shape[0] -1) # perplexity는 (1, n_samples) 사이여야 함
        
        if latent_vectors.shape[0] <= 1: # 최종 확인
            print("Cannot run TSNE with 1 or fewer samples after adjustments. Skipping TSNE.")
            return

        # n_iter를 max_iter로 변경, learning_rate='auto' 와 init='pca' 추가 (최신 권장 사항)
        reducer = TSNE(n_components=2, random_state=42, 
                       perplexity=perplexity_val, 
                       max_iter=1000, # 반복 횟수를 좀 더 늘려도 좋습니다 (기본값 1000)
                       init='pca',     # PCA로 초기화하면 더 안정적이고 빠른 수렴
                       learning_rate='auto') # 자동으로 학습률 조정
    elif method == 'umap':
        # UMAP의 n_neighbors도 샘플 수보다 작아야 함
        n_neighbors_val = min(15, latent_vectors.shape[0] - 1)
        if n_neighbors_val <=0:
            if latent_vectors.shape[0] <=1:
                print("Cannot run UMAP with 1 or fewer samples. Skipping UMAP.")
                return
            n_neighbors_val = max(1, latent_vectors.shape[0] // 2 -1)
            if n_neighbors_val == 0 and latent_vectors.shape[0] > 1:
                n_neighbors_val = latent_vectors.shape[0] - 1


        reducer = umap.UMAP(n_components=2, random_state=42, 
                            n_neighbors=n_neighbors_val, 
                            min_dist=0.1)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")
        
    embedding = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.7)
    
    plt.title(f'{method.upper()} Visualization of Latent Space (Latent Dim: {latent_dim})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{method}_latent_space_visualization.png')
    plt.savefig(save_path)
    print(f"{method.upper()} visualization saved to {save_path}")
    plt.close()


def main():
    cfg = EvaluationConfig()

    print(f"Using device: {DEVICE}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. 학습된 Autoencoder 모델 로드
    autoencoder, feature_dim, latent_dim = load_model_from_wandb_artifact(cfg)
    print(f"Autoencoder loaded. Input_dim from ViT: {feature_dim}, Latent_dim: {latent_dim}")

    # 2. Pretrained Theia 모델 로드
    print(f"Loading pretrained Theia model: {cfg.model_name}")
    pretrained_model_config = prepare_theia_transformer_model(model_name=cfg.model_name, model_device=DEVICE)
    pretrained_model = pretrained_model_config['model']()
    pretrained_inference_fn = pretrained_model_config['inference']

    # 3. 데이터 로더 준비 (검증 데이터셋)
    # train.py와 동일한 방식으로 데이터셋을 로드하고 분할합니다.
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data', cfg.camera_index, 'rgb')
    if not os.path.exists(data_root):
        print(f"Data root directory not found: {data_root}")
        print("Please ensure the 'data' directory is structured correctly relative to this script,")
        print("or update the 'data_root' path.")
        return

    full_dataloader = get_image_dataloader(
        data_root=data_root,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        batch_size=cfg.batch_size, # 재구성 평가는 배치 크기 그대로, 시각화는 어차피 샘플링
        normalize=False, # Theia 추론 함수 내에서 정규화 수행
        num_workers=os.cpu_count() // 2 or 1,
        shuffle=False # 평가 시에는 섞을 필요 없음 (재현성을 위해)
    )
    full_dataset = full_dataloader.dataset

    val_size = int(cfg.validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size # 사용하지 않지만, 분할을 위해 필요

    # random_split을 사용하면 매번 다른 인덱스로 분할될 수 있음.
    # 평가의 일관성을 위해, 고정된 시드 사용 또는 학습 시 저장된 인덱스 사용 필요.
    # 여기서는 간단히 random_split 사용 (매 실행 시 검증셋이 달라질 수 있음)
    # 좀 더 엄밀하게는 학습 시 사용한 검증셋 인덱스를 저장했다가 로드해야 합니다.
    # 여기서는 학습 스크립트의 random_split과 동일한 방식으로 데이터를 분할.
    # torch.manual_seed(42) # 데이터 분할의 일관성을 위해 시드 고정 가능
    # _, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 여기서는 검증 데이터셋 전체를 사용하거나, 일부를 샘플링해서 사용
    # SubsetRandomSampler 대신 전체 validation set을 순차적으로 사용하도록 수정
    indices = list(range(len(full_dataset)))
    # 학습 시와 동일한 방식으로 split 하기 위해, shuffle 후 split
    np.random.seed(42) # shuffle 에 대한 시드 고정 (train.py에서 random_split 내부적으로 하는 방식과 유사)
    np.random.shuffle(indices)
    val_indices = indices[train_size:] # train_size 이후가 val_set

    # val_sampler = SubsetRandomSampler(val_indices) # SubsetRandomSampler를 쓰면 shuffle=False여도 섞임
    # DataLoader는 dataset 객체와 sampler를 받거나, dataset과 shuffle 옵션을 받음
    # Subset으로 만들어서 DataLoader에 전달
    from torch.utils.data import Subset
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Using validation set of size: {len(val_dataset)}")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, # 평가 시 순서 중요할 수 있음 (또는 재현성)
        num_workers=os.cpu_count() // 2 or 1, 
        pin_memory=True
    )
    
    if len(val_loader) == 0:
        print("Validation loader is empty. Check dataset size and validation_split.")
        return

    # --- 2. 최종 재구성 성능 지표 ---
    print("\n--- Evaluating Reconstruction Performance ---")
    avg_mse, avg_cosine_sim = evaluate_reconstruction(
        autoencoder, pretrained_model, pretrained_inference_fn, val_loader, DEVICE
    )
    print(f"  Average MSE per sample on validation set: {avg_mse:.6f}")
    print(f"  Average Cosine Similarity on validation set: {avg_cosine_sim:.6f}")
    
    with open(os.path.join(cfg.output_dir, "reconstruction_metrics.txt"), "w") as f:
        f.write(f"Validation Set Results for Autoencoder (Run ID: {cfg.wandb_run_id}):\n")
        f.write(f"  Average MSE per sample: {avg_mse:.6f}\n")
        f.write(f"  Average Cosine Similarity: {avg_cosine_sim:.6f}\n")

    # --- 3. 잠재 공간 시각화 ---
    print("\n--- Visualizing Latent Space ---")
    # 시각화에는 데이터 로더를 다시 만들어서 shuffle=True로 하는 것이 다양한 샘플을 보는데 유리할 수 있음
    # 또는 현재 val_loader에서 num_samples_for_visualization 만큼만 사용
    vis_loader = DataLoader( # 시각화용 데이터 로더 (필요시 더 작은 배치 또는 셔플링)
        val_dataset, # 이미 분할된 검증 데이터셋 사용
        batch_size=cfg.batch_size, # 시각화시 배치 크기는 크게 중요하지 않음
        shuffle=True, # 다양한 샘플을 보기 위해 셔플링
        num_workers=os.cpu_count() // 2 or 1
    )

    if len(vis_loader.dataset) < cfg.num_samples_for_visualization:
        print(f"Warning: Requested {cfg.num_samples_for_visualization} samples for visualization, but validation set only has {len(vis_loader.dataset)} samples. Using all available samples.")
        num_vis_samples = len(vis_loader.dataset)
    else:
        num_vis_samples = cfg.num_samples_for_visualization
        
    if num_vis_samples > 0 :
        visualize_latent_space(
            autoencoder, pretrained_model, pretrained_inference_fn, vis_loader, DEVICE, 
            'tsne', num_vis_samples, cfg.output_dir, latent_dim
        )
        visualize_latent_space(
            autoencoder, pretrained_model, pretrained_inference_fn, vis_loader, DEVICE, 
            'umap', num_vis_samples, cfg.output_dir, latent_dim
        )
    else:
        print("Not enough samples in validation set to perform visualization.")

    print(f"\nEvaluation complete. Results saved in '{cfg.output_dir}' directory.")

if __name__ == "__main__":
    main()