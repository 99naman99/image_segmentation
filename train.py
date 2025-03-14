import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
from segmentation import UNET
from dataset import CarvanaDataset



if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 4
    EPOCHS = 3
    TRAIN_IMG_DIR = r"C:\Users\naman\Desktop\segmentation\carvana-image-masking-challenge\train1\train"
    TRAIN_MASK_DIR = r"C:\Users\naman\Desktop\segmentation\carvana-image-masking-challenge\train1\train_masks"
    MODEL_SAVE_PATH = r"C:\Users\naman\Desktop\segmentation\trainedmodel.pth"

    train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(TRAIN_IMG_DIR,TRAIN_MASK_DIR,transform=train_transform)

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)


    

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    model = UNET(in_channels=3,out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)