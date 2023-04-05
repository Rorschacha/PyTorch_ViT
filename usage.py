import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def inspect_img(**kwargs):
    """path=...,image=...,plt版,接受bgr图片/数组 或者文件路径"""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    path = kwargs.get('path')
    image = kwargs.get('image')
    show = kwargs.get('show')
    if path is not None:
        path = kwargs['path']
        image = mpimg.imread(path)
    elif image is not None:
        image = kwargs['image']

    if image is not None:
        print('  shape :', image.shape)
        print('   size :', image.size)
        if show:
            plt.imshow(image)
            plt.show()

    return image


def load_model_pth(model,model_path=None):
    # Load model parameters
    #model_path = "model_epoch_10.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()


# Prepare the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms(image).unsqueeze(0)

# Perform prediction on the input image
def predict(image_path, model):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)

        if True: #简单处理一下输出
            print('output:',output)
            probabilities = torch.softmax(output, dim=1)
    return probabilities


def usage():
    from models.vit import ViT
    model = ViT(
        image_size=224,
        patch_size=8,
        num_classes=2,
        dim=1024,
        depth=16,
        heads=16,
        mlp_dim=4096,
        dropout=0.1,
        emb_dropout=0.1
    )
    #    image_size = 256, # 其他尺寸：384
    #    patch_size = 32, # 切出来小片的边长，必须被image_size 整除
    #    num_classes = 1000, # 分类数量
    #    dim = 1024, # encoder前 线性层 输出的 矢量的长度
    #   depth = 6,  # transformer block的数量
    #   heads = 16, # Number of heads in Multi-head Attention layer
    #    mlp_dim = 2048,# MLP层缩放参数

    model_path = "pre_model_epoch_40.pth"
    load_model_pth(model, model_path)

    # Example usage
    image_path = r"D:\datasets\PracticeSets\cats and dogs\data\test\1.jpg"
    probabilities = predict(image_path, model)
    print(f"Probabilities: {probabilities}")

    # Get the class index with the highest probability
    predicted_class_index = torch.argmax(probabilities).item()
    print(f"Predicted class index: {predicted_class_index}")

    if True:
        inspect_img(path=image_path,show=True)




def main():
    usage()


if __name__ == '__main__':
    main()