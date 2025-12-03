import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Un bloc résiduel simple : y = f(x) + x
    Cela permet d'entraîner des réseaux plus profonds sans perdre l'information.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Si on change la taille (stride=2) ou le nombre de filtres,
        # on doit adapter l'entrée x pour pouvoir l'additionner à la sortie (skip connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # (Skip Connection)
        out = F.relu(out)
        return out

class CustomMultiHeadCNN(nn.Module):
    def __init__(self, n_color=5, n_length=3, dropout=0.3):
        super(CustomMultiHeadCNN, self).__init__()

        # --- 1. BACKBONE (Extracteur de features) ---

        # Entrée: 64x64x3
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Blocs Résiduels (Feature learning)
        # 64x64 -> 32x32
        self.layer1 = ResidualBlock(64, 64, stride=2)
        # 32x32 -> 16x16
        self.layer2 = ResidualBlock(64, 128, stride=2)
        # 16x16 -> 8x8
        self.layer3 = ResidualBlock(128, 256, stride=2)
        # 8x8 -> 4x4
        self.layer4 = ResidualBlock(256, 512, stride=2)

        # Global Pooling: Transforme n'importe quelle taille (4x4x512) en un vecteur (1x1x512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dimension du vecteur de features (sortie de layer4)
        self.feature_dim = 512

        # --- 2. HEADS (Classifieurs Spécifiques) ---
        # Chaque tête a sa propre petite couche cachée pour se spécialiser

        self.head_beard = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.head_mustache = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.head_glasses = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.head_color = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_color)
        )

        self.head_length = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_length)
        )

        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Passage dans le Backbone
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling
        features = self.global_pool(x)

        # Passage dans les Têtes
        return {
            "beard": self.head_beard(features).squeeze(1),
            "mustache": self.head_mustache(features).squeeze(1),
            "glasses": self.head_glasses(features).squeeze(1),
            "hair_color": self.head_color(features),
            "hair_length": self.head_length(features)
        }