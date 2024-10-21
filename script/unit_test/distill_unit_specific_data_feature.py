# import autorootcwd
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from segment_anything_feature import sam_model_registry
# from tiny_vit_sam_feature import TinyViT
# from copy import deepcopy
# from torch.utils.data import DataLoader
# import cv2
# from sklearn.decomposition import PCA

# # 기본 설정
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Dataset 클래스 정의 (Inference 용)
# class NpyDatasetInference(torch.utils.data.Dataset):
#     def __init__(self, data_root, npy_file):
#         self.data_root = data_root
#         self.img_path = os.path.join(data_root, 'imgs')
#         self.npy_file = npy_file
#         self.img_name = npy_file.split('_')[0] + '_' + npy_file.split('_')[1] + '.npy'

#     def __len__(self):
#         return 1  # 단일 데이터 추론

#     def __getitem__(self, index):
#         img_3c = np.load(os.path.join(self.img_path, self.img_name), 'r', allow_pickle=True)  # (H, W, 3)
#         img_256 = cv2.resize(img_3c, (256, 256), interpolation=cv2.INTER_AREA)
#         img_256 = np.transpose(img_256, (2, 0, 1))  # (3, H, W)
#         img_256 = (img_256 - img_256.min()) / np.clip(img_256.max() - img_256.min(), a_min=1e-8, a_max=None)

#         img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_AREA)
#         img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, H, W)
#         img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
#         return torch.tensor(img_256).float(),torch.tensor(img_1024).float(), self.img_name


# def inference_on_npy(data_root, npy_file=None, medsam_checkpoint=None, tinyvit_checkpoint=None):
#     # MedSAM teacher 모델 설정
#     medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
#     teacher_model = deepcopy(medsam_model.image_encoder).to(device)
#     teacher_model.eval()

#     # TinyViT student 모델 설정
#     student_model = TinyViT(
#         img_size=256,
#         in_chans=3,
#         embed_dims=[64, 128, 160, 320],
#         depths=[2, 2, 6, 2],
#         num_heads=[2, 4, 5, 10],
#         window_sizes=[7, 7, 14, 7],
#         mlp_ratio=4.0,
#         drop_rate=0.0,
#         drop_path_rate=0.0,
#         use_checkpoint=False,
#         mbconv_expand_ratio=4.0,
#         local_conv_size=3,
#         layer_lr_decay=0.8
#     ).to(device)

#     # 미리 학습된 TinyViT 모델 불러오기
#     if tinyvit_checkpoint and os.path.isfile(tinyvit_checkpoint):
#         print(f"Loading TinyViT model from checkpoint: {tinyvit_checkpoint}")
#         checkpoint = torch.load(tinyvit_checkpoint, map_location=device)
#         student_model.load_state_dict(checkpoint, strict=True)
#     else:
#         print("No TinyViT checkpoint found, using untrained model.")

#     student_model.eval()

#     # 데이터셋 생성
#     dataset = NpyDatasetInference(data_root, npy_file)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     # 단일 데이터 추론
#     for step, (image_256,image_1024, img_name) in enumerate(dataloader):
#         image_256 = image_256.to(device)

#         with torch.no_grad():
#             # Teacher model에서 feature 추출
#             teacher_output = teacher_model(image_1024)  # Teacher 모델 추론
#             # Student model에서 feature 추출
#             student_output = student_model(image_256)  # Student 모델 추론

#         # feature 저장 디렉토리 생성
#         output_dir = "teacher_student_features"
#         os.makedirs(output_dir, exist_ok=True)

#         # teacher model의 feature 저장
#         teacher_output_path = os.path.join(output_dir, f"{img_name[0]}_teacher_output.npy")
#         teacher_output_np = teacher_output.cpu().numpy()
#         np.save(teacher_output_path, teacher_output_np)
#         print(f"Teacher model feature saved at: {teacher_output_path}")

#         # student model의 feature 저장
#         student_output_path = os.path.join(output_dir, f"{img_name[0]}_student_output.npy")
#         student_output_np = student_output.cpu().numpy()
#         np.save(student_output_path, student_output_np)
#         print(f"Student model feature saved at: {student_output_path}")

#         break  # 한 번만 실행 후 종료





# if __name__ == "__main__":
#     data_root = "/mnt/sda/minkyukim/sam_dataset_refined/brats_npy_train_dataset_256image"
#     npy_file = "1_T1ce_lbl3.npy"
#     medsam_checkpoint = "/home/minkyukim/sam-tinyViT/work_dir/SAM/sam_vit_b_01ec64.pth"
#     tinyvit_checkpoint = "/mnt/sda/minkyukim/pth/tiny-brats-distill/tiny_model_best.pth"#"/home/minkyukim/sam-tinyViT/work_dir/MedSAM/tinyvit_pretrained.pt"

#     inference_on_npy(data_root, npy_file=npy_file, medsam_checkpoint=medsam_checkpoint, tinyvit_checkpoint=tinyvit_checkpoint)
import autorootcwd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from segment_anything_feature import sam_model_registry
from tiny_vit_sam_feature import TinyViT
from copy import deepcopy
from torch.utils.data import DataLoader
import cv2
from sklearn.decomposition import PCA

# 기본 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Dataset 클래스 정의 (Inference 용)
class NpyDatasetInference(torch.utils.data.Dataset):
    def __init__(self, data_root, npy_file):
        self.data_root = data_root
        self.img_path = os.path.join(data_root, 'imgs')
        self.npy_file = npy_file
        self.img_name = npy_file.split('_')[0] + '_' + npy_file.split('_')[1] + '.npy'

    def __len__(self):
        return 1  # 단일 데이터 추론

    def __getitem__(self, index):
        img_3c = np.load(os.path.join(self.img_path, self.img_name), 'r', allow_pickle=True)  # (H, W, 3)
        resize_img_skimg = cv2.resize(
            img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC
        )
        resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(resize_img_skimg_01, (2, 0, 1))
        assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'

        img_256 = cv2.resize(
            img_3c, (256, 256), interpolation=cv2.INTER_AREA
        )
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256 = np.transpose(img_256, (2, 0, 1))
        assert np.max(img_256)<=1.0 and np.min(img_256)>=0.0, 'image should be normalized to [0, 1]'

        return torch.tensor(img_1024).float(), torch.tensor(img_256).float(), self.img_name


def inference_on_npy(data_root, npy_file=None, medsam_checkpoint=None, tinyvit_checkpoint=None):
    # MedSAM teacher 모델 설정
    medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
    teacher_model = deepcopy(medsam_model.image_encoder).to(device)
    teacher_model.eval()

    # TinyViT student 모델 설정
    student_model = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    ).to(device)

    # 미리 학습된 TinyViT 모델 불러오기
    if tinyvit_checkpoint and os.path.isfile(tinyvit_checkpoint):
        print(f"Loading TinyViT model from checkpoint: {tinyvit_checkpoint}")
        checkpoint = torch.load(tinyvit_checkpoint, map_location=device)
        student_model.load_state_dict(checkpoint, strict=True)
    else:
        print("No TinyViT checkpoint found, using untrained model.")

    student_model.eval()

    # 데이터셋 생성
    dataset = NpyDatasetInference(data_root, npy_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 단일 데이터 추론
    for step, (image_1024, image_256, img_name) in enumerate(dataloader):
        image_256 = image_256.to(device)
        image_1024 = image_1024.to(device)

        with torch.no_grad():

            teacher_output = teacher_model(image_1024) 
        
            student_output = student_model(image_256)  
        # feature 저장 디렉토리 생성
        output_dir = "/home/minkyukim/sam-tinyViT/feature_output/ivdm_default"
        os.makedirs(output_dir, exist_ok=True)

        # teacher model의 feature 저장
        teacher_output_path = os.path.join(output_dir, f"{img_name[0]}_teacher_output.npy")
        teacher_output_np = teacher_output.cpu().numpy()
        np.save(teacher_output_path, teacher_output_np)
        print(f"Teacher model feature saved at: {teacher_output_path}")

        # student model의 feature 저장
        student_output_path = os.path.join(output_dir, f"{img_name[0]}_student_output.npy")
        student_output_np = student_output.cpu().numpy()
        np.save(student_output_path, student_output_np)
        print(f"Student model feature saved at: {student_output_path}")

        break  # 한 번만 실행 후 종료


if __name__ == "__main__":
    data_root = "/mnt/sda/minkyukim/sam_dataset_refined/ivdm_npy_train_dataset_256image"
    npy_file ="01-15_fat_3.npy" # "1_T1ce_lbl3.npy"
    medsam_checkpoint = "/home/minkyukim/sam-tinyViT/work_dir/SAM/sam_vit_b_01ec64.pth"
    tinyvit_checkpoint = "/mnt/sda/minkyukim/pth/tiny-ivdm-distill/tiny_model_20.pth"

    inference_on_npy(data_root, npy_file=npy_file, medsam_checkpoint=medsam_checkpoint, tinyvit_checkpoint=tinyvit_checkpoint)



