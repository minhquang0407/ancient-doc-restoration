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
import torch
import torch.nn as nn

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

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256) # Kết thúc Encoder cũ ở 256
        self.pool = nn.MaxPool2d(2)

        # --- BOTTLENECK (Thêm mới) ---
        # Đáy chữ U: Nơi chứa thông tin đặc trưng sâu nhất
        #self.bottleneck = DoubleConv(256, 512)
        self.bottleneck = SpectralTransform(256, 512)
        # --- Decoder ---
        # 0. Up từ Bottleneck (512) lên x4 (256)
        self.up0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(512, 256) # 256 từ up0 + 256 từ x4 = 512 vào -> ra 256

        # 1. Các lớp Decoder cũ (Giữ nguyên logic)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) # 128 + 128 = 256 vào -> ra 128

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)  # 64 + 64 = 128 vào -> ra 64

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)   # 32 + 32 = 64 vào -> ra 32

        self.outc = nn.Conv2d(32, self.n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.inc(x)            # 32

        x2 = self.pool(x1)
        x2 = self.down1(x2)         # 64

        x3 = self.pool(x2)
        x3 = self.down2(x3)         # 128

        x4 = self.pool(x3)
        x4 = self.down3(x4)         # 256

        # --- Bottleneck Path (Mới) ---
        x5 = self.pool(x4)          # Xuống sâu thêm 1 tầng
        x5 = self.bottleneck(x5)    # Xử lý tại đáy (512 kênh)

        # --- Decoder Path ---
        # Bắt đầu đi lên từ đáy
        x = self.up0(x5)                    # Up từ 512 -> 256
        x = torch.cat([x, x4], dim=1)       # Ghép với x4 (256)
        x = self.conv_up0(x)                # Conv trộn lại -> ra 256

        # Tiếp tục đi lên các tầng cũ
        x = self.up1(x)                     # Up từ 256 -> 128
        x = torch.cat([x, x3], dim=1)       # Ghép với x3 (128)
        x = self.conv_up1(x)

        x = self.up2(x)                     # Up từ 128 -> 64
        x = torch.cat([x, x2], dim=1)       # Ghép với x2 (64)
        x = self.conv_up2(x)

        x = self.up3(x)                     # Up từ 64 -> 32
        x = torch.cat([x, x1], dim=1)       # Ghép với x1 (32)
        x = self.conv_up3(x)

        logits = self.outc(x)

        return logits