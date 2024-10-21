import autorootcwd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from script.feature_python.train_brats import NpyDataset, show_mask, show_box, MedSAM_Lite, device, args, join
from segment_anything_feature.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam_feature import TinyViT

def inference_on_npy(data_root, npy_file=None, bbox_shift=20):

    medsam_lite_image_encoder = TinyViT(
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
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder
    ).to(device)


    medsam_lite_checkpoint = "/mnt/sda/minkyukim/pth/tiny-brats-distill/tiny_model_best.pth"
    if medsam_lite_checkpoint and os.path.isfile(medsam_lite_checkpoint):
        print(f"Loading model from checkpoint: {medsam_lite_checkpoint}")
        checkpoint = torch.load(medsam_lite_checkpoint, map_location=device)
        medsam_lite_model.load_state_dict(checkpoint, strict=True)
    else:
        print("No checkpoint found, using untrained model.")

    medsam_lite_model.eval()

    # 데이터셋 로드
    dataset = NpyDataset(data_root, bbox_shift=bbox_shift)
    if npy_file:
        dataset.gt_path_files = [join(dataset.gt_path, npy_file)]
        print(f"Running inference on specific npy file: {npy_file}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 데이터 추론 및 시각화
    for step, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        gt = batch["gt2D"].to(device)
        bboxes = batch["bboxes"].to(device)
        img_name = batch["image_name"][0]

        with torch.no_grad():
            pred_mask, _ = medsam_lite_model(image, bboxes)


        os.makedirs("gt_images", exist_ok=True)


        _, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 원본 이미지 시각화
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)
        axs[0].imshow(image_np)
        show_box(bboxes[0].cpu().numpy().squeeze(), axs[0])
        axs[0].set_title("Original Image with BBox")

        # GT 시각화
        gt_resized = F.interpolate(gt.float(), size=(256, 256), mode='nearest').squeeze(1)
        gt_resized_np = gt_resized.cpu().numpy().squeeze(0)
        axs[1].imshow(image_np)
        show_mask(gt_resized_np, axs[1])
        axs[1].set_title("GT Mask")

        # 예측된 마스크 시각화
        pred_mask_resized = F.interpolate(pred_mask.float(), size=(256, 256), mode='nearest').squeeze(1)
        pred_mask_resized_np = pred_mask_resized[0].cpu().numpy()
        pred_mask_resized_np = np.clip(pred_mask_resized_np, 0, 1)
        axs[2].imshow(image_np)
        show_mask(pred_mask_resized_np, axs[2])
        axs[2].set_title("Predicted Mask")

        # 결과 저장
        os.makedirs("output_images", exist_ok=True)
        plt.savefig(f"output_images/distill_output_{img_name}.png", bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Inference completed for: {img_name}")
        break


if __name__ == "__main__":
    data_root = "/mnt/sda/minkyukim/sam_dataset_refined/ivdm_npy_train_dataset_256image"
    npy_file = "01-15_fat_3.npy"#"1_T1ce_lbl3.npy"  "01-15_fat_3.npy"
    inference_on_npy(data_root, npy_file=npy_file)



