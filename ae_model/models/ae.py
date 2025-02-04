import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse

from geometry import (
    relaxed_distortion_measure,
    get_pullbacked_Riemannian_metric,
    get_flattening_scores
)

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

class VAE(AE):
    def __init__(
        self, encoder, decoder
    ):
        super(VAE, self).__init__(encoder, decoder)

    def encode(self, x):
        z = self.encoder(x)
        if len(z.size()) == 4:
            z = z.squeeze(2).squeeze(2)
        half_chan = int(z.shape[1] / 2)
        return z[:, :half_chan]

    def decode(self, z):
        return self.decoder(z)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        
        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)

        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            # "nll_": nll.item(),
            # "kl_loss_": kl_loss.mean(),
            # "sigma_": self.decoder.sigma.item(),
        }

    def eval_step(self, dl, **kwargs):
        device = kwargs["device"]
        score = []
        for x, _ in dl:
            z = self.encode(x.to(device))
            G = get_pullbacked_Riemannian_metric(self.decode, z)
            score.append(get_flattening_scores(G, mode='condition_number'))
        mean_condition_number = torch.cat(score).mean()
        return {
            "MCN_": mean_condition_number.item()
        }

    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]

        ## original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10
        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        recon = self.decode(self.encode(x.to(device)))
        x_img = make_grid(x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)
        recon_img = make_grid(recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1)

        # 2d graph (latent sapce)
        num_points_for_each_class = 200
        num_G_plots_for_each_class = 20
        label_unique = torch.unique(dl.dataset.targets)
        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        for label in label_unique:
            temp_data = dl.dataset.data[dl.dataset.targets == label][:num_points_for_each_class]
            temp_z = self.encode(temp_data.to(device))
            z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
            G = get_pullbacked_Riemannian_metric(self.decode, z_sampled)

            z_.append(temp_z)
            label_.append(label.repeat(temp_z.size(0)))
            z_sampled_.append(z_sampled)
            label_sampled_.append(label.repeat(z_sampled.size(0)))
            G_.append(G)


        z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
        label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
        color_ = label_to_color(label_)
        G_ = torch.cat(G_, dim=0).detach().cpu()
        z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
        label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
        color_sampled_ = label_to_color(label_sampled_)

        f = plt.figure()
        plt.title('Latent space embeddings with equidistant ellipses')
        z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
        eig_mean = torch.svd(G_).S.mean().item()
        scale = 0.1 * z_scale * np.sqrt(eig_mean)
        alpha = 0.3
        for idx in range(len(z_sampled_)):
            e = PD_metric_to_ellipse(np.linalg.inv(G_[idx,:,:]), z_sampled_[idx,:], scale, fc=color_sampled_[idx,:]/255.0, alpha=alpha)
            plt.gca().add_artist(e)
        for label in label_unique:
            label = label.item()
            plt.scatter(z_[label_==label,0], z_[label_==label,1], c=color_[label_==label]/255, label=label)
        plt.legend()
        plt.axis('equal')
        plt.close()
        f_np = np.transpose(figure_to_array(f), (2, 0, 1))[:3,:,:]

        return {
            'input@': torch.clip(x_img, min=0, max=1),
            'recon@': torch.clip(recon_img, min=0, max=1),
            'latent_space@': f_np
        }

class IRVAE(VAE):
    def __init__(
        self, encoder, decoder, iso_reg=1.0, metric='identity', 
    ):
        super(IRVAE, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        self.metric = metric
    
    def train_step(self, x, optimizer, **kwargs):
        # print(x.shape)
        # raise NotImplementedError
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        
        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        iso_loss = relaxed_distortion_measure(self.decode, z_sample, eta=0.2, metric=self.metric)
          
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}

class IRVAE_scs(VAE):
    def __init__(
        self, encoder, decoder, iso_reg=1.0, metric='identity', 
    ):
        super(IRVAE_scs, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        self.metric = metric
        self.pole_fc = nn.Linear(52*14, 33)
    
    def train_step(self, x_list, optimizer, **kwargs):
        # pole = x_list[:, :33]
        x = x_list[:, 33:].reshape(len(x_list),1, 52, 14)
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)

        # half_chan = int(z.shape[1] / 2)
        # mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        # rec_x = self.decode(mu).reshape(z.shape[0], -1)
        # # print(rec_x.shape)
        # rec_pole = self.pole_fc(rec_x)
        # ls = torch.nn.MSELoss(reduce=False) 

        # pole_loss = ls(pole, rec_pole)
        # pole_loss[:32] = pole_loss[:32]*10
        # pole_loss = pole_loss.mean()

        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        iso_loss = relaxed_distortion_measure(self.decode, z_sample, eta=0.2, metric=self.metric)
          
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}

    def eval_step(self, dl, **kwargs):
        device = kwargs["device"]
        score = []
        show_pole = False
        
        
        for x_list, _ in dl:
            pole = x_list[:, :33]
            x = x_list[:, 33:].reshape(len(x_list),1, 52, 14)
            z = self.encode(x.to(device))
            # if not show_pole:
            #     show_idx = np.random.randint(len(x))
            #     rec_x = self.decode(z).reshape(z.shape[0], -1)
            #     rec_pole = self.pole_fc(rec_x)
            #     print(f'origin {[int(p.item()) if i<32 else p.item() for i, p in enumerate(pole[show_idx])]}')
            #     print(f'recons {[round(p.item()+1)-1 if i<32 else p.item() for i, p in enumerate(rec_pole[show_idx])]}')
            #     show_pole = True
            G = get_pullbacked_Riemannian_metric(self.decode, z)
            score.append(get_flattening_scores(G, mode='condition_number'))

        mean_condition_number = torch.cat(score).mean()
        return {
            "MCN_": mean_condition_number.item()
        }
    
    def visual_step(self, x_list,logdir, **kwargs):
        x = x_list[:, 33:].reshape(x_list.shape[0], 1, 52, 14)
        recon = self(x)
        show_idx = np.random.randint(len(x))
        ori_y = x[show_idx].detach().cpu().numpy().reshape(52, 14)
        rec_y = recon[show_idx].detach().cpu().numpy().reshape(52, 14)

        plt.figure()
        plt.subplot(121)            
        plt.imshow(ori_y)
        plt.subplot(122)
        plt.imshow(rec_y)

        plt.savefig(f'{logdir}best_recon.png')
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}
    
    def validation_step(self, x_list, **kwargs):
        # pole = x_list[:, :33]
        x = x_list[:, 33:].reshape(x_list.shape[0], 1, 52, 14)
        recon = self(x)
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        # rec_x = self.decode(mu).reshape(z.shape[0], -1)
        # print(rec_x.shape)
        # rec_pole = self.pole_fc(rec_x)
        # ls = torch.nn.MSELoss(reduce=False) 

        # pole_loss = ls(pole, rec_pole)
        # pole_loss[:32] = pole_loss[:32]*10
        # pole_loss = pole_loss.mean()

        nll = - self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        iso_loss = relaxed_distortion_measure(self.decode, z_sample, eta=0.2, metric=self.metric)
          
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss
        # loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}
        
class IRVAE_scs_res(VAE):
    def __init__(
        self, encoder, decoder, iso_reg=1.0, metric='identity', 
    ):
        super(IRVAE_scs_res, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        self.metric = metric
    
    def loss_fn(recon_y, y, recon_p, p, code):
        ls = torch.nn.MSELoss() 

        l2_norm = torch.norm(code, dim=1).mean()
        map_mse = ls(recon_y,y)
        pole_mse = ls(recon_p, p)
        return map_mse, pole_mse, 0.001*l2_norm

    def train_step(self, x_list, optimizer, **kwargs):
        pole = x_list[:, :33]
        x = x_list[:, 33:].reshape(len(x_list),1, 52, 14)
        optimizer.zero_grad()
        

        code = self.encoder(x)

        code_dim = int(np.sqrt(code.shape[-1]))

        code = torch.reshape(code,(-1, code_dim, code_dim))
        code = torch.unsqueeze(code,axis=1)
        # recon_map, recon_pattern, recon_amp = self.decoder(code)
        recon_map, recon_p = self.decoder(code)

        recon_map = torch.unsqueeze(recon_map,axis=1)

        def loss_fn(recon_y, y, recon_p, p, code):
            ls = torch.nn.MSELoss() 

            l2_norm = torch.norm(code, dim=1).mean()
            map_mse = ls(recon_y,y)
            pole_mse = ls(recon_p, p)
            return map_mse, pole_mse, 0.001*l2_norm

        map_loss, pole_loss, reg_loss = loss_fn(recon_map, x,recon_p,pole, code)

        
        iso_loss = relaxed_distortion_measure(self.decode, z_sample, eta=0.2, metric=self.metric)
          
        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item(), "pole_loss": pole_loss.item()}

    def eval_step(self, dl, **kwargs):
        device = kwargs["device"]
        score = []
        show_pole = False
        
        
        for x_list, _ in dl:
            pole = x_list[:, :33]
            x = x_list[:, 33:].reshape(len(x_list),1, 52, 14)
            z = self.encode(x.to(device))
            if not show_pole:
                show_idx = np.random.randint(len(x))
                rec_x = self.decode(z).reshape(z.shape[0], -1)
                rec_pole = self.pole_fc(rec_x)
                print(f'origin {[int(p.item()) if i<32 else p.item() for i, p in enumerate(pole[show_idx])]}')
                print(f'recons {[round(p.item()+1)-1 if i<32 else p.item() for i, p in enumerate(rec_pole[show_idx])]}')
                show_pole = True
            G = get_pullbacked_Riemannian_metric(self.decode, z)
            score.append(get_flattening_scores(G, mode='condition_number'))

        mean_condition_number = torch.cat(score).mean()
        return {
            "MCN_": mean_condition_number.item()
        }
    
    def validation_step(self, x_list, **kwargs):
        x = x_list[:, 33:].reshape(x_list.shape[0], 1, 52, 14)
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}
        