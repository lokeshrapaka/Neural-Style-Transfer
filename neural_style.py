import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# --- DEVICE & IMAGE SIZE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 512 if torch.cuda.is_available() else 256

# --- IMAGE LOADER ---
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

def image_loader(path):
    image = Image.open(path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# --- DISPLAY IMAGE ---
def imshow(tensor, title=None):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

# --- PATHS ---
content_path = "content.png"
style_path = "style.png"

assert os.path.exists(content_path), f"❌ File not found: {content_path}"
assert os.path.exists(style_path), f"❌ File not found: {style_path}"

# --- LOAD IMAGES ---
content_img = image_loader(content_path)
style_img = image_loader(style_path)

assert content_img.size() == style_img.size(), "❗ Images must be the same size!"

# --- CONTENT LOSS ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# --- GRAM MATRIX & STYLE LOSS ---
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# --- NORMALIZATION LAYER ---
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# --- MODEL & LOSS CONSTRUCTION ---
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j+1]
    return model, style_losses, content_losses

# --- INPUT INIT ---
input_img = content_img.clone()
input_img.requires_grad_(True)

# --- BUILD MODEL ---
model, style_losses, content_losses = get_style_model_and_losses(
    cnn, cnn_mean, cnn_std, style_img, content_img)

optimizer = optim.LBFGS([input_img])

# --- STYLE TRANSFER ---
print("🔄 Optimizing style transfer...")

num_steps = 300
for step in range(num_steps):
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * 1e6 + content_score
        loss.backward()
        if step % 50 == 0:
            print(f"Step {step}: Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")
        return loss
    optimizer.step(closure)

# --- SHOW FINAL IMAGE ---
input_img.data.clamp_(0, 1)
print("✅ Style transfer complete! Displaying result...")
imshow(input_img, title="Stylized Image")
