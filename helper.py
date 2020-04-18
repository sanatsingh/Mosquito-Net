import io
import torch
import numpy as np
from model_st import ( MosquitoNet , MosquitoNet_Mish )
from PIL import Image
import torchvision.transforms as transforms

def get_model():
    # Model's state dictionary is loaded, Save model state_dict after training using : torch.save(model.state_dict(), PATH)
    checkpoint_path = 'saved_model_patha_to_go_here'
    # Replace MosquitoNet with MosquitoNet_Mish to use the mish version. Model to be trained seperately for that. 
    model = MosquitoNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def get_tensor(image_bytes):
    t_trans = transforms.Compose([
        transforms.Resize((120,120)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(io.BytesIO(image_bytes))
    return t_trans(img)

def pixel_sim(img_a):
    ''' Pixel Wise Image Comparison for filtering out Garbage inputs '''
    t_trans = transforms.Compose([
        transforms.Resize((120,120)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_a = img_a.numpy()
    img_b = t_trans(Image.open('static/images/d.png')).numpy()

    return np.sum(np.absolute(img_a - img_b)) / (120*120)