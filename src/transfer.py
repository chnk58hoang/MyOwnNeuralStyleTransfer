import torch
import argparse
import numpy as np
from utils import *
from torch.optim import LBFGS
from torch.autograd import Variable
from torch.optim import LBFGS
from torch import nn


def get_losses(model,target_content,target_styles,optimized_img,mse_loss_content,mse_loss_style,content_layer,style_layers,weights):
    
    pred_content = model(optimized_img)[content_layer[0]].squeeze(0)
    pred_styles = [model(optimized_img)[i] for i in style_layers[0]]

    pred_styles = [gram_mattrix(i) for i in pred_styles]
    
    content_loss = mse_loss_content(target_content,pred_content)
    style_loss = 0.0
    for pred_gram,target_gram in zip(pred_styles,target_styles):
        style_loss += mse_loss_style(target_gram[0],pred_gram[0])
    style_loss /= len(target_styles)
    variation_loss = total_variation(optimized_img)
    total_loss = weights['content'] * content_loss + weights['style'] * style_loss + weights['variation'] * variation_loss
    return total_loss,content_loss, style_loss,variation_loss



def optimizing_step(optimizer,model,target_content,target_styles,optimized_img,mse_loss_content,mse_loss_style,content_layer,style_layers,weights):
    cnt = 0
    def closure():
        nonlocal cnt
        total_loss,content_loss,style_loss,variation_loss = get_losses(model,target_content,target_styles,optimized_img,mse_loss_content,mse_loss_style,content_layer,style_layers,weights)
        print(f'Step:{cnt}. Total_loss: {total_loss:.3f}. Content_loss: {content_loss:.3f}. Style_loss: {style_loss:.3f}. Variation_loss: {variation_loss:.3f}')
        optimizer.zero_grad()
        total_loss.backward()
        cnt += 1
        return total_loss
    optimizer.step(closure)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img',type=str,default='tubingen.png')
    parser.add_argument('--style_img',type=str,default='wave_crop.jpg')
    parser.add_argument('--save_path',type=str,default='save_image.png')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    "Load image"
    content_img = transform_image(image_path=args.content_img,device=device,target_shape=None)
    style_img = transform_image(image_path=args.style_img,device=device,target_shape=np.asarray(content_img.shape[2:]))

    "init optimized_img"
    init_img = content_img
    optimized_img = Variable(init_img,requires_grad=True)


    "Load model"
    model, content_feature_index_name, style_feature_indices_names = get_model(device)

    "Get target content and style feature representations"
    target_content_feature_map = model(content_img)[content_feature_index_name[0]].squeeze(0)
    style_img_fms = [model(style_img)[i] for i in style_feature_indices_names[0]]
    
    "Convert target style representations to gram mattrix"
    target_style_feature_maps = [gram_mattrix(i) for i in style_img_fms]

    "Loss weights"
    weights = {'content':1e5,'style':3e4,'variation':1e0}

    "Define optimizer"
    optimizer = LBFGS((optimized_img,),max_iter=1000,line_search_fn='strong_wolfe')

    "Define loss"
    mse_loss_content = nn.MSELoss(reduction='mean')
    mse_loss_style = nn.MSELoss(reduction='sum')


    # "Optimize process"
    # optimizing_step(optimizer=optimizer,model=model,target_content=target_content_feature_map,
    #                     target_styles=target_style_feature_maps,optimized_img=optimized_img,mse_loss_content=mse_loss_content,
    #                     mse_loss_style = mse_loss_style,content_layer=content_feature_index_name,style_layers=style_feature_indices_names,weights=weights)
        


    "Save generated image"
    save_image(optimized_img,args.save_path)

    
    