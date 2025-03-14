import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from dataset import CarvanaDataset
from segmentation import UNET





def single_image_inference(image_pth, model_pth, device):
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()])
    model.eval()

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = torch.sigmoid(model(img))

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask = (pred_mask > 0.5).float()

    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img)
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = r"C:\Users\naman\Desktop\segmentation\carvana-image-masking-challenge\29bb3ece3180_11.jpg"
    TRAIN_IMG_DIR = r"C:\Users\naman\Desktop\segmentation\carvana-image-masking-challenge\train1\check"
    TRAIN_MASK_DIR = r"C:\Users\naman\Desktop\segmentation\carvana-image-masking-challenge\train1\check_mask"
    MODEL_PATH = r"C:\Users\naman\Desktop\segmentation\trainedmodel.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
