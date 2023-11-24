from model import VGG19_Model
from torchvision import transforms
import numpy as np
import torch
import cv2 as cv
import os


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def load_image(img_path, target_shape=None):
    """Load image from file path

    Args:
        img_path: (str) path to the image
        target_shape: (int|tuple) new shape of the image we want to resize
    """

    # Read image
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    # Limit the upper bound of the image shape to ensure model work
    if img.shape[0] > 800:
        target_shape = 800

    # Resize section
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def transform_image(image_path, device, target_shape):
    """ Transform image to torch.Tensor

    Args:
        img_path: (str) path to the image
        target_shape: (int|tuple) new shape of the image we want to resize
        device: (str) device name
    """

    # Load image
    img = load_image(image_path, target_shape)

    # Image transformation
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255)),
                                    transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
                                    ])

    img = transform(img=img).to(device).unsqueeze(0)
    return img


def save_image(optimized_img, image_path):
    """Save image to file

    Args:
        optimized_img: (tensor) tensor represent the result image
        image_path: (str) path to save location

    """
    optimized_img = optimized_img.squeeze(0).to('cpu').detach().numpy()
    out_img = np.moveaxis(optimized_img, 0, 2)
    out_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    out_img = np.clip(out_img, 0, 255).astype('uint8')
    cv.imwrite(filename=image_path, img=out_img[:, :, ::-1])


def get_model(device):
    """Load the NST model

    Args:
        device: (str) device name
    """

    # Load model to device
    model = VGG19_Model()
    model.to(device).eval()

    # Get content and style layers infor
    layer_names = model.layer_names
    content_feature_map_index = model.content_feature_map_index
    style_feature_map_indices = model.style_feature_maps_indices
    content_feature_index_name = (content_feature_map_index, layer_names[content_feature_map_index])
    style_layer_names = layer_names[:content_feature_map_index] + layer_names[content_feature_map_index + 1:]
    style_feature_indices_names = (style_feature_map_indices, style_layer_names)

    return model, content_feature_index_name, style_feature_indices_names


def gram_mattrix(x, normalize=True):
    """Compute the Gram matrix of a tensor"""

    batch, channel, height, width = x.size()
    feature = x.view(batch, channel, -1)
    feature_t = feature.transpose(1, 2)
    gram = torch.bmm(feature, feature_t)

    if normalize:
        gram = gram / (channel * width * height)
    return gram


def init_target_image(image):
    """Init the result image the same size as another image

    Args:
        image: (tensor)
    """
    init_image = torch.rand_like(image)
    return init_image


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
