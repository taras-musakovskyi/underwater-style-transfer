import torch
import torch.nn as nn
from PIL import Image
from cog import BasePredictor, Input, Path
from typing import Literal
import random
import torchvision.transforms as T

# ============ Pix2Pix UNet Generator ============
class UNetDown(nn.Module):
    """Downsampling block with Conv-BN-LeakyReLU"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """Upsampling block with ConvTranspose-BN-ReLU"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], 1)
        return x

class UNetGenerator(nn.Module):
    """UNet Generator with skip connections"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

# ============ CycleGAN ResNet Generator ============
class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetGenerator(nn.Module):
    """ResNet Generator for CycleGAN (512×512×3 RGB)"""
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling (512 → 256 → 128)
        model += [
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]

        # Residual blocks (128×128×256)
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(256)]

        # Upsampling (128 → 256 → 512)
        model += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()  # Output in [-1, 1] range
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ============ Predictor ============
class Predictor(BasePredictor):
    def setup(self):
        """Load models at container startup"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")

        # Load Pix2Pix generator
        print("📦 Loading Pix2Pix generator (epoch 90)...")
        self.pix2pix_gen = UNetGenerator(in_channels=3, out_channels=3).to(self.device)
        checkpoint = torch.load(
            "/src/models/pix2pix_generator_epoch90.pth",
            map_location=self.device
        )
        self.pix2pix_gen.load_state_dict(checkpoint)
        self.pix2pix_gen.eval()
        print("✅ Pix2Pix loaded")

        # Load CycleGAN generator
        print("📦 Loading CycleGAN generator (best checkpoint)...")
        self.cyclegan_gen = ResNetGenerator(in_channels=3, out_channels=3).to(self.device)
        cyclegan_checkpoint = torch.load(
            "/src/models/cyclegan_best_dry2wet.pth",
            map_location=self.device
        )
        self.cyclegan_gen.load_state_dict(cyclegan_checkpoint)
        self.cyclegan_gen.eval()
        print("✅ CycleGAN loaded")

    def predict(
        self,
        image: Path = Input(description="Input apartment/room image (supports JPG, PNG, HEIC, WebP)"),
        model: Literal["pix2pix", "cyclegan"] = Input(
            description="Model to use",
            default="pix2pix"
        ),
        alpha: float = Input(
            description="Effect strength (0=original, 1=full underwater). Only for Pix2Pix",
            default=0.75,
            ge=0.0,
            le=1.0
        ),
        seed: int = Input(
            description="Random seed (-1 for random)",
            default=-1
        )
    ) -> Path:
        """Apply underwater style transfer"""

        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)

        print(f"🎨 Processing with {model}, alpha={alpha}, seed={seed}")

        if model == "pix2pix":
            return self._run_pix2pix(image, alpha)
        elif model == "cyclegan":
            return self._run_cyclegan(image)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _run_pix2pix(self, image_path: Path, alpha: float) -> Path:
        """Run Pix2Pix with alpha blending"""
        print("🔄 Running Pix2Pix generator...")

        # Load and preprocess (supports HEIC via pillow-heif)
        img = Image.open(str(image_path)).convert("RGB")
        original_size = img.size

        # Resize to 512×512 for model
        img_resized = img.resize((512, 512), Image.LANCZOS)

        # To tensor, normalize to [-1, 1]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)

        # Generate
        with torch.no_grad():
            generated = self.pix2pix_gen(img_tensor)

        # Denormalize [-1, 1] → [0, 1]
        generated = (generated + 1) / 2
        generated = generated.squeeze(0).cpu()

        # To PIL
        to_pil = T.ToPILImage()
        generated_img = to_pil(generated)

        # Resize back to original size
        generated_img = generated_img.resize(original_size, Image.LANCZOS)
        img_original_size = img

        # Alpha blending
        if alpha < 1.0:
            print(f"🎨 Alpha blending: {alpha:.2f}")
            blended = Image.blend(img_original_size, generated_img, alpha)
            result_img = blended
        else:
            result_img = generated_img

        # Save output
        output_path = "/tmp/output.jpg"
        result_img.save(output_path, "JPEG", quality=95)
        print(f"✅ Saved result: {result_img.size}")

        return Path(output_path)

    def _run_cyclegan(self, image_path: Path) -> Path:
        """Run CycleGAN generator"""
        print("🔄 Running CycleGAN generator...")

        # Load and preprocess (supports HEIC via pillow-heif)
        img = Image.open(str(image_path)).convert("RGB")
        original_size = img.size

        # Resize to 512×512 for model
        img_resized = img.resize((512, 512), Image.LANCZOS)

        # To tensor, normalize to [-1, 1]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)

        # Generate
        with torch.no_grad():
            generated = self.cyclegan_gen(img_tensor)

        # Denormalize [-1, 1] → [0, 1]
        generated = (generated + 1) / 2
        generated = generated.squeeze(0).cpu()

        # To PIL
        to_pil = T.ToPILImage()
        generated_img = to_pil(generated)

        # Resize back to original size
        generated_img = generated_img.resize(original_size, Image.LANCZOS)

        # Save output
        output_path = "/tmp/output.jpg"
        generated_img.save(output_path, "JPEG", quality=95)
        print(f"✅ Saved result: {generated_img.size}")

        return Path(output_path)
