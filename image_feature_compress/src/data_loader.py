import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_image_dataloader(
    data_root: str,
    image_height: int,
    image_width: int,
    batch_size: int,
    num_channels: int = 3,  # RGB 이미지의 경우 3
    normalize: bool = True, # 정규화 여부를 선택하는 인자 추가
    num_workers: int = None,
    shuffle: bool = True
) -> DataLoader:
    """
    지정된 경로에서 이미지를 로드하고 전처리하여 DataLoader를 반환합니다.
    이 DataLoader는 이미지를 (0~1) 텐서로 변환하고 크기를 조절하며,
    선택적으로 (0.5, 0.5) 정규화를 통해 [-1, 1] 범위로 변환합니다.

    Args:
        data_root (str): 이미지가 포함된 데이터셋의 루트 디렉토리 경로.
                         예: `data_root/category_1/image1.jpg`
        image_height (int): 이미지의 높이 (픽셀).
        image_width (int): 이미지의 너비 (픽셀).
        num_channels (int): 이미지 채널 수 (RGB: 3). 기본값은 3.
        batch_size (int): DataLoader가 반환할 배치 크기.
        normalize (bool, optional): 이미지를 (0.5, 0.5)로 정규화하여 [-1, 1] 범위로 만들지 여부.
                                    기본값은 True. False인 경우 [0, 1] 범위로 유지됩니다.
        num_workers (int, optional): 데이터를 로드할 때 사용할 서브프로세스 수.
                                    기본값은 CPU 코어 수의 절반 또는 1.
        shuffle (bool, optional): 에포크마다 데이터를 섞을지 여부. 기본값은 True.

    Returns:
        torch.utils.data.DataLoader: 이미지 데이터셋을 로드하고 배치 처리하는 DataLoader 객체.

    Raises:
        Exception: 데이터 로딩에 실패할 경우 (경로, 권한 등 문제).
    """
    if num_workers is None:
        num_workers = os.cpu_count() // 2 or 1

    transform_list = [
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(), # 이미지를 [0, 1] 범위로 스케일링
    ]

    if normalize:
        # 이미지를 (0.5, 0.5)로 정규화하여 [-1, 1] 범위로 만듭니다.
        transform_list.append(
            transforms.Normalize(
                (0.5,) * num_channels,
                (0.5,) * num_channels,
            )
        )
    
    transform = transforms.Compose(transform_list)

    try:
        image_dataset = datasets.ImageFolder(root=data_root, transform=transform)
        
        if len(image_dataset) == 0:
            print(f"Warning: No images found in {data_root}. Please check the directory.")
            print("Expected structure: your_image_dataset/some_category_folder/image.jpg")

        image_dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Successfully loaded {len(image_dataset)} images from {data_root}.")
        return image_dataloader

    except Exception as e:
        print(f"Error loading images from {data_root}. Please check the directory structure and permissions.")
        print(f"Error details: {e}")
        print("Expected structure: your_image_dataset/some_category_folder/image.jpg")
        raise