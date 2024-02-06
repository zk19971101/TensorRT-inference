import torch
import cv2 as cv

from torchvision.models import resnet34, ResNet34_Weights


def save_trace_model(model, model_name):
    model.eval()
    trace_script_module = torch.jit.trace(model, torch.rand(1, 3, 512, 512))
    trace_script_module.save(model_name)
    print("save trace model success!")


def load_trace_model(model_name):
    model = torch.jit.load(model_name)
    out = model(torch.rand(1, 3, 512, 512))
    print(out.shape)
    return model


def test_resnet():
    img = cv.imread('./test.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img / 255., dtype=torch.float)
    print(img_tensor.shape)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = (img_tensor - mean) / std
    inputs = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    print("tensor", inputs.shape)
    print(inputs[0, :, 0, 0])
    weight = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weight)
    res = model(inputs).softmax(1).argmax(1).squeeze(0).detach().numpy()
    print(res.shape)


if __name__ == '__main__':
    model_name = "./asset/model.pt"
    weight = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weight)
    save_trace_model(model, model_name)
    para_path = "asset/para.pth"
    torch.save(model.state_dict(), para_path)
    torch.jit.load(model_name)
    load_trace_model(model_name)