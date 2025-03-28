import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from torchvision import transforms
import torchvision.models as models

# 文件路徑
train_path = "./app/brain_mri/train"
test_path = "./app/brain_mri/test"
val_path = "./app/brain_mri/valid"

# COCO 標註檔案路徑（針對每個資料夾）
train_annotations_file = os.path.join(train_path, '_annotations.coco.json')
test_annotations_file = os.path.join(test_path, '_annotations.coco.json')
val_annotations_file = os.path.join(val_path, '_annotations.coco.json')

# 讀取 COCO 格式的標註檔案
def load_coco_annotations(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

# 根據 image_id 獲取圖片的標註
def get_annotations_for_image(coco_data, image_id):
    annotations = coco_data['annotations']
    return [ann for ann in annotations if ann['image_id'] == image_id]

# 根據標註的 category_id 獲取標註對應的類別名稱
def get_category_name_by_id(coco_data, category_id):
    categories = coco_data['categories']
    category_name = next((cat['name'] for cat in categories if cat['id'] == category_id), None)
    return category_name

# 根據 image_id 生成對應的 mask 路徑
def get_mask_for_image(coco_data, image_id, image_shape):
    annotations = get_annotations_for_image(coco_data, image_id)
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
    # 生成遮罩（使用 bounding box）
    for ann in annotations:
        category_name = get_category_name_by_id(coco_data, ann['category_id'])
        if category_name == 'Tumor':  # 假設 "Tumor" 是我們的目標類別
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = map(int, bbox)
            mask[y:y+h, x:x+w] = 255  # 將遮罩設為白色

    return mask

# ResUNet++ 模型實現
class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNetPlusPlus, self).__init__()
        # 使用預訓練的 ResNet50 作為編碼器
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Identity()  # 去除 ResNet 的分類層
        
        # 定義解碼器部分：多層反卷積
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),  # 由2048映射至1024
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),   # 由1024映射至512
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),    # 由512映射至256
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),    # 由256映射至128
            nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2)  # 最後映射至輸出尺寸
        )
        
    def forward(self, x):
        # 編碼器部分：ResNet50
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        # 解碼器部分：將特徵圖通過轉置卷積還原到輸出大小
        x = self.decoder(x)
        return x



# 數據集處理
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, coco_data, transform=None):
        self.image_paths = image_paths
        self.coco_data = coco_data
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 讀取圖片
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        
        # 根據 image_id 生成對應的遮罩
        image_id = idx  # 假設每個圖片的 id 是它在資料夾中的位置
        mask = get_mask_for_image(self.coco_data, image_id, image.shape)

        # 增加預處理
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Unsqueeze to match the expected dimension

        return torch.tensor(image, dtype=torch.float32), mask


# 數據預處理
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),  # 設定為64x64以匹配模型輸出
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



# 修改的 load_annotations 函數，使用 os.listdir 而非 glob
def load_image_paths(path, file_extension=".jpg"):
    # 遍歷文件夾並過濾指定擴展名的圖片
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(file_extension)]
    return image_paths


def create_dataloader(image_paths, coco_data, batch_size=32):
    transform = get_transform()
    dataset = SegmentationDataset(image_paths, coco_data, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 訓練循環
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloaders['train']):.4f}")


# 測試循環
def evaluate_model(model, dataloaders):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            preds = outputs > 0.5

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).ravel()
    y_pred = np.concatenate(y_pred, axis=0).ravel()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Background", "Tumor"]))


# 主函數
def main():
    # 載入訓練資料集的 COCO 標註檔案
    coco_data_train = load_coco_annotations(train_annotations_file)

    # 載入訓練與測試數據集
    train_image_paths = load_image_paths(train_path)
    test_image_paths = load_image_paths(test_path)

    # 使用 DataLoader 創建訓練和測試集
    train_loader = create_dataloader(train_image_paths, coco_data_train)
    test_loader = create_dataloader(test_image_paths, coco_data_train)

    # 初始化模型、損失函數和優化器
    model = ResUNetPlusPlus().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 訓練模型
    train_model(model, {'train': train_loader}, criterion, optimizer)

    # 測試模型
    evaluate_model(model, {'test': test_loader})


if __name__ == "__main__":
    main()
