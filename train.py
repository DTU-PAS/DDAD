import torch
import torchvision
import os
import torch.nn as nn
from forward_process import *
from dataset import *

from torch.optim import Adam
from dataset import *
from torchvision import models
import timm
from backbone import *



def build_optimizer(model, config):
    lr = config.model.learning_rate
    weight_decay = config.model.weight_decay
    return Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

def get_loss(model, constant_dict, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t , constant_dict['sqrt_alphas_cumprod'], constant_dict['sqrt_one_minus_alphas_cumprod'], device)
    noise_pred = model(x_noisy, t)
    loss = F.l1_loss(noise, noise_pred)
    #loss = F.mse_loss(noise, noise_pred)

    return loss




@torch.no_grad()
def sample_timestep(config, model, constant_dict, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(constant_dict['betas'], t, x.shape)

    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        constant_dict['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(constant_dict['sqrt_recip_alphas'], t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(constant_dict['posterior_variance'], t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        # rng = numpy.random.default_rng()
        # ixr, iyr = rng.random(256)*1.5, rng.random(256)*1.5
        # noiser = opensimplex.noise2array(ixr, iyr)
        # noiser = torch.Tensor(noiser)

        # ixg, iyg =  rng.random(256)*1.5, rng.random(256)*1.5
        # noiseg = opensimplex.noise2array(ixg, iyg)
        # noiseg = torch.Tensor(noiseg)

        # ixb, iyb =  rng.random(256)*1.5, rng.random(256)*1.5
        # noiseb = opensimplex.noise2array(ixb, iyb)
        # noiseb = torch.Tensor(noiseb)

        # noise = torch.stack((noiser, noiseg, noiseb)).to(config.model.device)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 




@torch.no_grad()
def sample_plot_image(model, trainloader, constant_dict, epoch, config):
    image = next(iter(trainloader))[0]
    # Sample noise
    trajectoy_steps = torch.Tensor([config.model.test_trajectoy_steps]).type(torch.int64)

    image = forward_diffusion_sample(image, trajectoy_steps, constant_dict['sqrt_alphas_cumprod'], constant_dict['sqrt_one_minus_alphas_cumprod'], device=config.model.device)[0]
    num_images = 10
    trajectory_steps = config.model.trajectory_steps
    stepsize = int(trajectory_steps/num_images)
    
    
    plt.figure(figsize=(15,15))
    plt.axis('off')
    image_to_show =show_tensor_image(image)
    plt.subplot(1, num_images+1, int(trajectory_steps/stepsize)+1)
    plt.imshow(image_to_show)
    plt.title(trajectory_steps)
    for i in range(0,trajectory_steps-1)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        image = sample_timestep(config, model, constant_dict, image, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images+1, int(i/stepsize)+1)
            image_to_show =show_tensor_image(image.detach().cpu())
            plt.imshow(image_to_show)
            plt.title(i)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('results/backward_process_after_{}_epochs.png'.format(epoch))
    # plt.show()





def trainer(model, constant_dict, config):
    optimizer = build_optimizer(model, config)
    train_dataset = MVTecDataset(
        root= config.data.data_dir,
        category=config.data.category,
        input_size= config.data.image_size,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )

    


   # feature_extractor = Feature_extractor(config)
  #  feature_extractor.to(config.model.device)

    for epoch in range(config.model.epochs):
        for step, batch in enumerate(trainloader):
            

        #    features = feature_extractor(batch[0].to(config.model.device))
            optimizer.zero_grad()
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            loss = get_loss(model, constant_dict, batch[0], t) 
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item()}  |  Memory allocated : {torch.cuda.memory_allocated(config.model.device)}")
            if epoch %100 == 0 and step ==0:
                sample_plot_image(model, trainloader, constant_dict, epoch, config)


    if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(config.model.checkpoint_dir, config.model.checkpoint_name),
    )

