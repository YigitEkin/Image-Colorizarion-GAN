import matplotlib.pyplot as plt
import torch

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def plot_results(loss_meter_dict, history):
    for name, loss_meter in loss_meter_dict.items():
        print(name)
        arr = [curr[name].cpu().detach().numpy() for curr in history]
        plt.plot([i for i in range(len(arr))], arr)
        plt.title(name)
        plt.show()
    
        

def update_losses(model, loss_meter_dict, count, history):
    curr = {}
    for loss_name, loss_meter in loss_meter_dict.items():
        curr[loss_name] = getattr(model, loss_name)
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)
        history.append(curr)
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.rgb
    grayscale = model.grayscale
    fake_imgs =  fake_color
    real_imgs = real_color
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(grayscale[i][0].cpu().numpy(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i].permute(1,2,0).cpu().numpy())
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i].permute(1,2,0).cpu().numpy())
        ax.axis("off")
    plt.show()
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")