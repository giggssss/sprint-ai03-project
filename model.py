import torch
import torch.nn as nn

# 간단한 YOLO11 구조 정의
class YOLO11(nn.Module):
    def __init__(self, num_classes=20, grid_size=7):
        super(YOLO11, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        # feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        # prediction head
        out_channels = (5 + num_classes)  # 5: [x, y, w, h, conf]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (self.grid_size // 16) * (self.grid_size // 16), 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, self.grid_size * self.grid_size * out_channels),
        )

    def forward(self, x):
        B = x.size(0)
        feat = self.backbone(x)
        pred = self.head(feat)
        return pred.view(B, self.grid_size, self.grid_size, 5 + self.num_classes)

# Model and loss only
class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        return self.mse(preds, targets)


# Example usage for YOLO11
if __name__ == "__main__":
    # Instantiate and print model architecture
    model = YOLO11()
    print(model)