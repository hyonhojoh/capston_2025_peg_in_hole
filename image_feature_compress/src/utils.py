import torch

def prepare_theia_transformer_model(model_name: str, model_device: str) -> dict:
    """Prepare the Theia transformer model for inference."""
    from transformers import AutoModel

    def _load_model() -> torch.nn.Module:
        """Load the Theia transformer model."""
        model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
        return model.to(model_device)

    def _inference(model, images: torch.Tensor) -> torch.Tensor:
        """Inference the Theia transformer model."""
        image_proc = images.to(model_device).float()
        mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
        image_proc = (image_proc - mean) / std

        with torch.no_grad(): # <-- 특징 추출 시에는 no_grad()를 사용하는 것이 안전하고 효율적입니다.
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
        
        # [수정] [CLS] 토큰 (인덱스 0)의 특징 벡터만 반환합니다.
        return features.last_hidden_state[:, 0]

    return {"model": _load_model, "inference": _inference}


def get_theia_features_shape(model, dummy_input_shape, device):
    """Theia 모델의 출력 특징 차원을 결정하는 헬퍼 함수."""
    dummy_images = torch.randn(dummy_input_shape).to(device)

    image_proc = dummy_images.float()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    image_proc = (image_proc - mean) / std

    with torch.no_grad():
        features_output = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
        
        # [수정] [CLS] 토큰 특징 벡터를 추출합니다.
        cls_features = features_output.last_hidden_state[:, 0]

    return cls_features.shape[1] # 배치 차원을 제외한 특징 차원 반환