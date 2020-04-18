import json
import torch
from helper import get_model, get_tensor, pixel_sim

model = get_model()
classes=['Infected','Uninfected']

def get_result(image_bytes):
    tensor = get_tensor(image_bytes)
    flag = pixel_sim(tensor)
    if(flag>1.3):
        return "Invalid Input"
    tensor = tensor.view(-1, 3, 120, 120)
    outputs = model(tensor)
    predicted = torch.max(outputs, 1)[1]
    prval = torch.max(outputs, 1)[0]
    result = classes[predicted]
    return  result
