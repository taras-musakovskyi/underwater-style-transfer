# Underwater Style Transfer

Transform apartment/room photos into underwater scenes using GANs.

## Models

### Pix2Pix (Available)
- **Architecture:** UNet Generator + PatchGAN Discriminator
- **Training:** 100 epochs on 1252 paired dry-wet apartment images
- **Checkpoint:** Epoch 90 (~230MB)
- **Features:** Alpha blending for intensity control (0.0-1.0)

### CycleGAN (Coming Soon)
- **Architecture:** ResNet Generators (9 residual blocks)
- **Status:** Training in progress

## Deployment

Deployed on Replicate via GitHub Actions:
- Push to `main` branch triggers build
- Model file baked into Docker image
- API: `taras-musakovskyi/underwater-style-transfer`

## Sample Images

6 HEIC sample images in `apt_samples/` for testing.

## Local Development

```bash
# Install Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog

# Test locally
cog predict -i image=@apt_samples/IMG_4352.HEIC -i model=pix2pix -i alpha=0.75

# Push to Replicate
cog login
cog push r8.im/taras-musakovskyi/underwater-style-transfer
```

## API Usage

```python
import replicate

output = replicate.run(
    "taras-musakovskyi/underwater-style-transfer",
    input={
        "image": open("apartment.jpg", "rb"),
        "model": "pix2pix",
        "alpha": 0.75,
        "seed": -1
    }
)
```

## Parameters

- **image**: Input apartment/room photo (JPG, PNG, HEIC, WebP)
- **model**: `"pix2pix"` or `"cyclegan"` (cyclegan coming soon)
- **alpha**: Effect strength 0.0-1.0 (Pix2Pix only, default 0.75)
- **seed**: Random seed, -1 for random

## Training Details

See `CLAUDE.md` in main deployment repo for full training history.
