import multiprocessing

import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import time
import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
class TextLineDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512)):
        """
        Args:
            root_dir (str): Đường dẫn đến folder dataset (chứa subfolder 'images' và 'masks').
            image_size (tuple): Kích thước resize (H, W).
        """
        self.root_dir = root_dir
        self.image_size = image_size

        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        # 1. Cấu hình các đuôi file chấp nhận
        self.valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

        # 2. Quét toàn bộ file ảnh hợp lệ
        self.filenames = []
        if os.path.exists(self.img_dir):
            all_files = os.listdir(self.img_dir)
            # Lọc file có đuôi nằm trong danh sách cho phép
            self.filenames = [f for f in all_files if os.path.splitext(f)[1].lower() in self.valid_exts]
            self.filenames.sort() # Sắp xếp để đảm bảo thứ tự cố định
        else:
            print(f"Lỗi: Không tìm thấy thư mục ảnh tại {self.img_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # --- BƯỚC 1: LẤY ĐƯỜNG DẪN ẢNH ---
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # --- BƯỚC 2: TÌM MASK TƯƠNG ỨNG (QUAN TRỌNG) ---
        # Logic: Ảnh tên "doc_01.jpg", thì mask có thể là "doc_01.png" hoặc "doc_01.bmp"
        # Ta phải tìm file mask có cùng tên cơ sở (basename)
        basename = os.path.splitext(img_name)[0]
        mask_path = None

        # Ưu tiên tìm .png trước (vì mask thường là png), sau đó đến các đuôi khác
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            potential_path = os.path.join(self.mask_dir, basename + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

        # Nếu vẫn không thấy mask, thử tìm chính xác tên file ảnh (trường hợp copy paste y chang)
        if mask_path is None:
             potential_path = os.path.join(self.mask_dir, img_name)
             if os.path.exists(potential_path):
                 mask_path = potential_path

        # --- BƯỚC 3: LOAD DỮ LIỆU ---
        # Load Grayscale (1 kênh màu)
        image = None
        mask = None

        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- BƯỚC 4: KIỂM TRA LỖI (Robustness) ---
        if image is None or mask is None:
            # print(f"Cảnh báo: Lỗi file tại index {idx}. Ảnh: {img_path}, Mask: {mask_path}. Đang bỏ qua...")
            # Đệ quy: Lấy mẫu kế tiếp để thế vào chỗ bị lỗi
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        # --- BƯỚC 5: RESIZE & CHUẨN HÓA ---
        # Resize về kích thước chuẩn (512x512) để đưa vào mạng
        image = cv2.resize(image, self.image_size)

        # Resize mask: Dùng Nearest Neighbor để giữ biên sắc nét (không bị mờ thành màu xám)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Chuẩn hóa về [0, 1] và float32
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Nhị phân hóa Mask tuyệt đối (Chỉ 0.0 và 1.0)
        # Ngưỡng 0.5: Dưới là nền, trên là dòng kẻ
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        # --- BƯỚC 6: CHUYỂN SANG TENSOR ---
        # PyTorch yêu cầu shape [Channel, Height, Width]
        # Ảnh đang là [H, W] -> unsqueeze(0) -> [1, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor
import torch
import torch.nn as nn
import torch.fft

class FourierUnit(nn.Module):
    """
    Bộ xử lý miền tần số: Chuyển sang phổ -> Conv -> Chuyển về ảnh.
    Giúp nắm bắt thông tin toàn cục (Global Context).
    """
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # 1. Fast Fourier Transform (Real -> Complex)
        # rfft2 trả về chỉ một nửa tần số (do tính đối xứng), tiết kiệm bộ nhớ
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            # 2. Xử lý trên miền tần số
            # Ghép phần thực và phần ảo lại để dùng Conv2d
            # Shape: (Batch, C, H, W/2 + 1) -> Tách thực/ảo -> (Batch, C*2, H, W/2 + 1)
            ffted = torch.fft.rfft2(x, norm='ortho')
            ffted = torch.cat([ffted.real, ffted.imag], dim=1)

            ffted = self.conv_layer(ffted)
            ffted = self.relu(self.bn(ffted))

            # 3. Tách thực ảo để Inverse FFT
            ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
            ffted_complex = torch.complex(ffted_real, ffted_imag)

            # 4. Inverse FFT (Complex -> Real)
            output = torch.fft.irfft2(ffted_complex, s=(h, w), norm='ortho')
        return output

class SpectralTransform(nn.Module):
    """
    Module FFC hoàn chỉnh: Chia luồng Local (Conv) và Global (Fourier)
    """
    def __init__(self, in_channels, out_channels, ratio=0.5):
        super(SpectralTransform, self).__init__()

        # Chia kênh: 50% đi đường Conv thường, 50% đi đường Fourier
        self.ratio = ratio
        mid_channels = int(in_channels * ratio)
        global_channels = in_channels - mid_channels

        # Nhánh Local (Giữ chi tiết cạnh, nét chữ)
        if mid_channels > 0:
            self.local_path = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.local_path = None # Tắt nhánh này nếu ratio = 0

        # --- 2. Nhánh Global (Chỉ tạo nếu cần) ---
        if global_channels > 0:
            self.global_path = FourierUnit(global_channels, global_channels)
        else:
            self.global_path = None # Tắt nhánh này nếu ratio = 1

        # Trộn lại (Fusion)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Chia kênh
        mid = int(x.shape[1] * self.ratio)
        x_local = x[:, :mid, :, :]
        x_global = x[:, mid:, :, :]

        # Xử lý song song
        out_local = self.local_path(x_local)
        out_global = self.global_path(x_global)

        # Ghép lại
        out = torch.cat([out_local, out_global], dim=1)
        return self.fusion(out)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)
class TinyUNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes=3):
        super(TinyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        #Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        #self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = SpectralTransform(256, 512)
        #Decoder
        self.up1 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.pool(x1)
        x2 = self.down1(x2)

        x3 = self.pool(x2)
        x3 = self.down2(x3)

        x4 = self.pool(x3)
        x4 = self.down3(x4)

        x4 = self.bottleneck(x4)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)

        return logits
def calculate_metrics_tensor(pred, target, threshold=0.5):
    """
    Tính IoU, F1-Score, Pixel Accuracy cùng lúc trên GPU để tối ưu tốc độ.
    """
    # 1. Binarize (Chuyển về 0 và 1)
    pred_mask = (torch.sigmoid(pred) > threshold).float()

    # Làm phẳng (Flatten) để tính toán vector
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    # --- TÍNH CÁC THÀNH PHẦN CƠ BẢN ---
    tp = (pred_flat * target_flat).sum() # True Positive
    fp = pred_flat.sum() - tp            # False Positive
    fn = target_flat.sum() - tp          # False Negative
    tn = torch.numel(pred_flat) - (tp + fp + fn) # True Negative

    epsilon = 1e-6

    # --- 1. IoU ---
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)

    # --- 2. F1-Score (Dice Coefficient) ---
    # F1 = 2*TP / (2*TP + FP + FN)
    f1 = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

    # --- 3. Pixel Accuracy (PA) ---
    # PA = (TP + TN) / Total
    pa = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)

    return iou.item(), f1.item(), pa.item()

def calculate_hd(pred, target, threshold=0.5):
    """
    Tính Hausdorff Distance (HD).
    Lưu ý: Phải chuyển sang CPU và dùng Scipy.
    Trả về khoảng cách (pixel). Càng THẤP càng tốt.
    """
    # Chuyển sang Numpy và Binarize
    pred_mask = (torch.sigmoid(pred) > threshold).detach().cpu().numpy()
    target_mask = target.detach().cpu().numpy()

    batch_hd = 0.0
    batch_size = pred_mask.shape[0]

    for i in range(batch_size):
        # Lấy toạ độ các điểm có giá trị 1 (Foreground)
        # shape [C, H, W] -> lấy [0, :, :] -> shape [H, W]
        p_coords = np.argwhere(pred_mask[i, 0] > 0)
        t_coords = np.argwhere(target_mask[i, 0] > 0)

        # Xử lý trường hợp Mask rỗng (tránh lỗi crash)
        if len(p_coords) == 0 or len(t_coords) == 0:
            # Nếu cả 2 đều rỗng -> Khoảng cách = 0 (Tuyệt vời)
            if len(p_coords) == 0 and len(t_coords) == 0:
                dist = 0.0
            # Nếu 1 trong 2 rỗng -> Phạt nặng (ví dụ lấy đường chéo ảnh)
            else:
                dist = 100.0 # Giá trị phạt tượng trưng (hoặc 512*sqrt(2))
        else:
            # Tính khoảng cách Hausdorff 2 chiều
            # directed_hausdorff(u, v) trả về (max(min(d(u,v))), index_u, index_v)
            d1 = directed_hausdorff(p_coords, t_coords)[0]
            d2 = directed_hausdorff(t_coords, p_coords)[0]
            dist = max(d1, d2)

        batch_hd += dist

    return batch_hd / batch_size

def bce_dice_loss(pred, target):
    # 1. Tính BCE Loss (Đảm bảo reduction='mean' để ra 1 con số)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')

    # 2. Tính Dice Loss
    pred_sigmoid = torch.sigmoid(pred)
    smooth = 1e-5

    # Ép phẳng (Flatten) ảnh về dạng [Batch_Size, -1] để tính toán an toàn hơn
    # Thay vì sum(dim=(2,3)), ta sum hết các pixel của mỗi ảnh
    pred_flat = pred_sigmoid.view(pred_sigmoid.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    # Dice cho từng ảnh trong batch
    dice_score = (2.0 * intersection + smooth) / (union + smooth)

    # Lấy trung bình cộng của cả batch để ra 1 con số (Scalar)
    dice_loss = 1.0 - dice_score.mean()

    # 3. Tổng hợp
    return 0.5 * bce + 0.5 * dice_loss
@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    # Khởi tạo các biến tổng
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    total_pa = 0
    total_hd = 0

    num_batches = len(loader)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # 1. Forward
        preds = model(images)

        # 2. Tính Loss
        loss = bce_dice_loss(preds, masks)
        total_loss += loss.item()

        # 3. Tính IoU, F1, PA (Nhanh - trên GPU)
        iou, f1, pa = calculate_metrics_tensor(preds, masks)
        total_iou += iou
        total_f1 += f1
        total_pa += pa

        # 4. Tính HD (Chậm - trên CPU)
        # Nếu thấy validate quá lâu, có thể comment dòng này lại
        #hd = calculate_hd(preds, masks)
        #total_hd += hd

    # Tính trung bình
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_f1 = total_f1 / num_batches
    avg_pa = total_pa / num_batches
    #avg_hd = total_hd / num_batches

    # Trả về Dictionary cho dễ quản lý
    metrics = {
        "loss": avg_loss,
        "iou": avg_iou,
        "f1": avg_f1,
        "pa": avg_pa,
        #"hd": avg_hd
    }

    return metrics

def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train() # Chế độ huấn luyện (bật Dropout/BatchNorm)
    epoch_loss = 0

    # Thanh tiến trình
    current_lr = optimizer.param_groups[0]['lr']
    loop = tqdm(loader, desc="Training", leave=False)

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast('cuda'):
        # 1. Forward (Chạy xuôi)
            preds = model(images)
            loss = bce_dice_loss(preds, masks)

        # 2. Backward (Chạy ngược - Tính đạo hàm)
         # Xóa gradient cũ
        scaler.scale(loss).backward()

        # 3. Update Weights (Cập nhật trọng số)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # Cập nhật thanh tiến trình
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item(), lr=current_lr)

    return epoch_loss / len(loader)
import random

# --- 1. HÀM KHÓA SEED CHUẨN MỰC ---
def seed_everything(seed=42):
    """
    Thiết lập seed cho toàn bộ hệ thống để đảm bảo kết quả tái lập được (Reproducibility).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Nếu dùng nhiều GPU

    # Đảm bảo tính nhất quán của thuật toán Convolution
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔒 Global Seed set to {seed}")
def main():
    torch.manual_seed(42)
    seed_everything(42)
    # 1. Setup Thiết bị (GPU hay CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    HYPERPARAMS = {
        'data_dir': 'dataset_dewarp_train',  # Folder data bạn vừa tạo
        'img_size': (512, 512),
        'batch_size': 16,          # Nếu GPU yếu thì giảm xuống 2 hoặc 1
        'learning_rate': 1e-3,
        'start_epoch': 0,
        'epochs': 50,             # Số vòng lặp huấn luyện
        'save_dir': 'assets',     # Nơi lưu model
        'model_name': 'tiny_unet_ffc_dewarp.pth',
        'num_workers': min(4, multiprocessing.cpu_count()),
        'log_name': 'training_log_ffc_.csv',
    }
    # 2. Chuẩn bị Dữ liệu
    print("Loading dataset...")
    full_dataset = TextLineDataset(HYPERPARAMS['data_dir'], HYPERPARAMS['img_size'])

    # Chia 90% Train - 10% Validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=HYPERPARAMS['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=HYPERPARAMS['batch_size'], shuffle=False, num_workers=8)

    print(f"   Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    # 3. Khởi tạo Model & Optimizer
    model = TinyUNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=HYPERPARAMS['learning_rate'], weight_decay=1e-4)

    # LR Scheduler: Giảm tốc độ học khi loss không giảm nữa (giúp hội tụ sâu hơn)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')
    # 4. Vòng lặp Huấn luyện
    os.makedirs(HYPERPARAMS['save_dir'], exist_ok=True)
    save_path = os.path.join(HYPERPARAMS['save_dir'], HYPERPARAMS['model_name'])
    log_csv_path = os.path.join(HYPERPARAMS['save_dir'], HYPERPARAMS['log_name'])
    best_val_loss = float('inf')
    if os.path.exists(save_path):
        try:
            print(f"🔄 Tìm thấy checkpoint: {save_path}. Đang thử load...")
            state = torch.load(save_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            best_val_loss = state['best_val_loss']
            HYPERPARAMS['start_epoch'] = state['epoch']
            print(f"✅ Resume training từ Epoch {state['epoch']}")
        except Exception as e:
            print(f"⚠️ Không load được checkpoint cũ (do khác kiến trúc hoặc file lỗi).")
            print(f"   Lỗi chi tiết: {e}")
            print("🚀 Sẽ Train mới từ đầu (Fresh Start)!")
    else:
        print("🚀 Khởi động Train mới hoàn toàn (Fresh Start)!")
    if os.path.exists(log_csv_path) and HYPERPARAMS['start_epoch'] > 0:
        history_df = pd.read_csv(log_csv_path)
    else:
        history_df = pd.DataFrame()
    start_time = time.time()
    print("Start Training...")
    for epoch in range(HYPERPARAMS['start_epoch'],HYPERPARAMS['epochs']):
        # Train
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)

        # Validate
        with torch.amp.autocast('cuda'):
            metrics = validate(model, val_loader, device)
        val_loss = metrics['loss']
        val_iou  = metrics['iou']
        val_f1   = metrics['f1']
        val_pa   = metrics['pa']
        #val_hd   = metrics['hd']
        # Update Scheduler

        scheduler.step(metrics['iou'])
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_f1': val_f1,
            'val_pa': val_pa,
            # 'val_hd': val_hd,
            'lr': current_lr,
            'time_sec': epoch_duration
        }

        history_df = pd.concat([history_df, pd.DataFrame([log_entry])], ignore_index=True)
        history_df.to_csv(log_csv_path, index=False)

        print(f"Epoch [{epoch+1}/{HYPERPARAMS['epochs']}] "
              f"T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f} | "
              f"IoU: {val_iou:.4f} | F1: {val_f1:.4f} | Pa: {val_pa:.4f} |"
              f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, save_path)
            print(f"----> New best model saved! (Loss: {best_val_loss:.4f})")
    total_time = time.time() - start_time
    print(f"Training Complete in {total_time/60:.1f} minutes!")
    print(f"Log saved to: {log_csv_path}")

main()