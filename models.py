"""
models/loss
"""
from networks import DynamicGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,ssim
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import sys


# x y is subject
# a b is illumination

class Models(nn.Module):
    def __init__(self, hyperparameters):
        super(Models, self).__init__()
        lr = hyperparameters['lr']
        self.model_name = hyperparameters['models_name']
        # Initiate the networks

        if(self.model_name=='dynamic_human'):
            self.gen = DynamicGen(hyperparameters['input_dim_a'], hyperparameters['gen'])
        else:
            sys.exit('error on models')

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim*2, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim*2, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen.parameters()) #+ list(self.gen_b.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))


        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_mask(self, input, target,mask):
        return torch.mean(torch.abs(torch.mul(input,mask) - torch.mul(target,mask)))

    def recon_criterion_rmse(self,input,target,mask,denorm=True):
        if(denorm):
            input = (input*0.5+0.5)
            target= (target*0.5+0.5)
        out=0
        psnr_=0
        ssim_=0
        if(len(input.shape)==3):
            tmp=torch.sum((torch.mul(input, mask) - torch.mul(target, mask))**2)
            tmp/=torch.sum(mask)
            tmp = tmp**0.5
            psnr = 20*torch.log10(1/tmp)
            img1 = torch.mul(input, mask) + torch.mul(target, 1-mask)
            img1 = torch.unsqueeze(img1,dim=0)
            img2 = torch.unsqueeze(target,dim=0)
            ssim_loss=ssim(img1, img2)
            #ssim_loss = pytorch_ssim.SSIM(window_size=11)
            return tmp.item(),psnr.item(),ssim_loss.item()
        else:
            for i in range(len(input)):
                tmp=torch.sum((torch.mul(input[i], mask[i]) - torch.mul(target[i], mask[i]))**2)
                tmp/=torch.sum(mask[i])
                tmp=tmp** 0.5
                out+=tmp
                psnr_+=20*torch.log10(1/tmp)

                img1 = torch.mul(input[i], mask[i]) + torch.mul(target[i], 1 - mask[i])
                img1 = torch.unsqueeze(img1, dim=0)
                img2 = torch.unsqueeze(target[i], dim=0)
                ssim_ +=ssim(img1, img2)

            return (out/len(input)).item(),(psnr_/len(input)).item(),(ssim_/len(input)).item()


    def test_recon_criterion_mask(self, input, target,mask):
        out=0
        if(len(input.shape)==3):
            tmp=torch.sum(torch.abs(torch.mul(input, mask) - torch.mul(target, mask)))
            tmp/=torch.sum(mask)
            return tmp
        else:
            for i in range(len(input)):
                tmp=torch.sum(torch.abs(torch.mul(input[i], mask[i]) - torch.mul(target[i], mask[i])))
                tmp/=torch.sum(mask[i])
                out+=tmp

            return out/len(input)


    def test_forward(self,x_a,y_b,gt_xb,x_mask,y_mask): ###
        # encode
        if (self.model_name != 'baseline'):
            c_x, s_a_prime,_,_ = self.gen.encode(x_a,x_mask)
            c_y, s_b_prime,_,_ = self.gen.encode(y_b,y_mask)
        else:
            c_x, s_a_prime = self.gen.encode(x_a,x_mask)
            c_y, s_b_prime = self.gen.encode(y_b,y_mask)
        # decode (within domain)
        x_a_recon =  torch.mul(self.gen.decode(c_x, s_a_prime,x_mask),x_mask) +  torch.mul(x_a,1-x_mask)
        x_b_recon_prime = torch.mul(self.gen.decode(c_x, s_b_prime,x_mask),x_mask)+torch.mul(gt_xb,1-x_mask)
        #self.loss_gen_recon_x_a = self.test_recon_criterion_mask(x_a_recon, x_a,x_mask)
        #self.loss_gen_prime_x_b = self.test_recon_criterion_mask(x_b_recon_prime, gt_xb,x_mask)

        xa_rmse,xa_psnr,xa_ssim=self.recon_criterion_rmse(x_a_recon, x_a,x_mask)
        xb_rmse, xb_psnr, xb_ssim = self.recon_criterion_rmse(x_b_recon_prime, gt_xb, x_mask)

        image_anchor = x_a[0:1].detach().cpu()[:3]
        image_recons = x_a_recon[0:1].detach().cpu()[:3]
        image_relight = x_b_recon_prime[0:1].detach().cpu()[:3]
        image_relight_target = y_b[0:1].detach().cpu()[:3]
        image_gt = gt_xb[0:1].detach().cpu()[:3]
        self.image_display = torch.cat((image_anchor, image_recons, image_relight, image_gt,image_relight_target), dim=3)
        return xa_rmse,xa_psnr,xa_ssim,xb_rmse, xb_psnr, xb_ssim

    def test_rotation(self,Xa_out, Yb_out, rotate, Xa_mask,Yb_mask):
        c_x, _, _, _ = self.gen.encode(Xa_out, Xa_mask)
        _, s_b, _, _ = self.gen.encode(Yb_out, Yb_mask)
        prev_adain_params, adain_params, next_adain_params  = self.gen._mlp_output(s_b)

        prev_prev, _=self.gen.dynamic_mlp.back_activate(prev_adain_params)
        _,next_next = self.gen.dynamic_mlp.back_activate(next_adain_params)

        #-90,-60,-30,0,30,60,90
        rmse_ = [0]*12
        psnr_ = [0]*12
        ssim_ = [0]*12
        out=[]
        delta_ = (prev_adain_params - prev_prev) / 3
        for i in range(0, 3):
            _params = prev_prev + delta_ * i
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)

            xb_rmse, xb_psnr, xb_ssim = self.recon_criterion_rmse(_image, rotate[i], Xa_mask)
            rmse_[i] += xb_rmse
            psnr_[i] += xb_psnr
            ssim_[i] += xb_ssim
            out.append(_image[0:1].detach().cpu()[:3])

        delta_ = (adain_params - prev_adain_params) / 3
        for i in range(3, 6):
            _params = prev_adain_params + delta_ * (i-3)
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            xb_rmse, xb_psnr, xb_ssim = self.recon_criterion_rmse(_image, rotate[i], Xa_mask)
            rmse_[i] += xb_rmse
            psnr_[i] += xb_psnr
            ssim_[i] += xb_ssim
            out.append(_image[0:1].detach().cpu()[:3])
        delta_ = (next_adain_params - adain_params) / 3
        for i in range(6, 9):
            _params = adain_params + delta_ * (i-6)
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            xb_rmse, xb_psnr, xb_ssim = self.recon_criterion_rmse(_image, rotate[i], Xa_mask)
            rmse_[i] += xb_rmse
            psnr_[i] += xb_psnr
            ssim_[i] += xb_ssim
            out.append(_image[0:1].detach().cpu()[:3])
        delta_ = (next_next - next_adain_params) / 3
        for i in range(9, 12):
            _params = next_adain_params + delta_ * (i-9)
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            xb_rmse, xb_psnr, xb_ssim = self.recon_criterion_rmse(_image, rotate[i], Xa_mask)
            rmse_[i] += xb_rmse
            psnr_[i] += xb_psnr
            ssim_[i] += xb_ssim
            out.append(_image[0:1].detach().cpu()[:3])
        return torch.tensor(rmse_),torch.tensor(psnr_),torch.tensor(ssim_),out


    def dynamic_gen_update(self, x_a,gt_xb, y_b,gt_xb_prev,gt_xb_next,  x_mask,y_mask,rand_y_out, rand_y_mask, hyperparameters):

        self.gen_opt.zero_grad()
        s_b_human_rand = Variable(torch.randn(y_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_x, s_a_prime, s_a_human, s_a_bg = self.gen.encode(x_a, x_mask)
        c_y, s_b_prime, s_b_human, s_b_bg = self.gen.encode(y_b, y_mask)
        # decode (within true)
        x_a_recon = self.gen.decode(c_x, s_a_prime, x_mask)


        # decode on prime
        x_b_recon_prime_prev, x_b_recon_prime, x_b_recon_prime_next, x_b_prev_adain_params, x_b_adain_params, x_b_next_adain_params \
            = self.gen.dynamic_decode(c_x, s_b_prime, x_mask)

        cross_xb = self.gen.decode(c_x, s_b_human_rand, x_mask)  # + torch.mul(gt_xb,1-x_mask)

        # create for latent cyclen loss
        c_x_recon, s_b_recon, _, _ = self.gen.encode(cross_xb, x_mask)

        # encode dynamics
        _, x_b_recon_prev_style, _, _ = self.gen.encode(gt_xb_prev, x_mask)
        x_b_recon_prev_prev_adain_params, x_b_recon_prev_adain_params, x_b_recon_prev_next_adain_params = self.gen._mlp_output(x_b_recon_prev_style)

        _, x_b_recon_next_style, _, _ = self.gen.encode(gt_xb_next, x_mask)
        x_b_recon_next_prev_adain_params, x_b_recon_next_adain_params, x_b_recon_next_next_adain_params = self.gen._mlp_output(x_b_recon_next_style)



        # image reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion_mask(x_a_recon, x_a, x_mask)
        # main relight loss
        self.loss_gen_prime_x_b = self.recon_criterion_mask(x_b_recon_prime, gt_xb, x_mask)


        # augmented relight
        self.loss_gen_prime_x_b_prev = self.recon_criterion_mask(x_b_recon_prime_prev, gt_xb_prev, x_mask)
        self.loss_gen_prime_x_b_next = self.recon_criterion_mask(x_b_recon_prime_next, gt_xb_next, x_mask)

        # feature cycle consistency
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon[:, :6], s_b_human_rand[:, :6])
        self.loss_gen_recon_c_x = self.recon_criterion(c_x_recon, c_x)

        # lighting code consistency:
        self.loss_gen_light_code_x_b = self.recon_criterion(x_b_recon_prev_ot_params, x_b_prev_ot_params) + \
                                       self.recon_criterion(x_b_recon_prev_next_ot_params, x_b_ot_params) + \
                                       self.recon_criterion(x_b_recon_next_prev_ot_params, x_b_ot_params) + \
                                       self.recon_criterion(x_b_recon_next_ot_params, x_b_next_ot_params) + \
                                       self.recon_criterion(x_b_recon_prev_prev_ot_params, x_b_recon_next_next_ot_params)


        self.loss_gen_total = hyperparameters['recon'] * self.loss_gen_recon_x_a + \
                              hyperparameters['relight'] * self.loss_gen_prime_x_b + \
                              hyperparameters['feat'] * self.loss_gen_recon_c_x + \
                              hyperparameters['feat'] * self.loss_gen_recon_s_b + \
                              hyperparameters['auglight'] * self.loss_gen_prime_x_b_prev + \
                              hyperparameters['auglight'] * self.loss_gen_prime_x_b_next + \
                              hyperparameters['cons'] * self.loss_gen_light_code_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

        image_anchor = x_a[0:1].detach().cpu()[:3]
        image_recons = x_a_recon[0:1].detach().cpu()[:3]

        image_x_b_recon_prev = x_b_recon_prime_prev[0:1].detach().cpu()[:3]
        image_x_b_recon_next = x_b_recon_prime_next[0:1].detach().cpu()[:3]
        image_relight = x_b_recon_prime[0:1].detach().cpu()[:3]
        image_gt = gt_xb[0:1].detach().cpu()[:3]
        image_yb = y_b[0:1].detach().cpu()[:3]

        self.image_display = torch.cat(
            (image_anchor, image_recons, image_x_b_recon_prev, image_relight, image_x_b_recon_next, image_gt, image_yb),
            dim=3)




    def update_learning_rate(self):
        # if self.dis_scheduler is not None:
        #     self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters,need_opt=True,path=None):
        # Load generators
        if(path==None):
            last_model_name = get_model_list(checkpoint_dir, "gen")
        else:
            last_model_name=path
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['a'])
        iterations = int(last_model_name[-11:-3])
        if(need_opt):
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_'+last_model_name[-11:-3]+'.pt'))
            self.gen_opt.load_state_dict(state_dict['gen'])
            self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer_%08d.pt'% (iterations + 1))
        torch.save({'a': self.gen.state_dict()}, gen_name)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)


    def forward_test_real(self,Xa,a_mask,ref_beauty=None,ref_mask=None,ref_bg=None,ref_img_full=None):
        c_x, s_s, _, _ = self.gen.encode(Xa, a_mask)
        if(ref_beauty is not None):
            c_ref,s_ref,_,_ = self.gen.encode(ref_beauty,ref_mask)

            # decode on prime
            x_b_recon_prime_prev, x_b_recon_prime, x_b_recon_prime_next, x_b_prev_ot_params, x_b_ot_params, x_b_next_ot_params \
                = self.gen.dynamic_decode(c_x, s_ref, a_mask)

            #output = self.gen.decode(c_x, s_ref)
            x_b_recon_prime_prev = torch.mul(x_b_recon_prime_prev, a_mask) + (-1) * (1 - a_mask)
            x_b_recon_prime_prev = x_b_recon_prime_prev[0:1].detach().cpu()[:3]

            x_b_recon_prime = torch.mul(x_b_recon_prime, a_mask) + (ref_bg) * (1 - a_mask) #ref_bg
            x_b_recon_prime = x_b_recon_prime[0:1].detach().cpu()[:3]

            x_b_recon_prime_next = torch.mul(x_b_recon_prime_next, a_mask) + (-1) * (1 - a_mask)
            x_b_recon_prime_next = x_b_recon_prime_next[0:1].detach().cpu()[:3]



            #ref_img = ref_beauty[0:1].detach().cpu()[:3]
            ref_img = ref_beauty[0:1].detach().cpu()[:3]

            output = torch.cat((Xa[0:1].detach().cpu()[:3], ref_img, x_b_recon_prime_prev,x_b_recon_prime,x_b_recon_prime_next),dim=3)
            #output = x_b_recon_prime

        else:

            output = self.gen.decode(c_x, s_s)
            output= torch.mul(output,a_mask)+(-1)*(1-a_mask)
            output = output[0:1].detach().cpu()[:3]
        return output



    def forward_test_interpolation(self,Xa,a_mask,ref_beauty,ref_mask, sample_num=4):###
        c_x, s_s,_,_ = self.gen.encode(Xa, a_mask)
        c_ref, s_ref,_,_  = self.gen.encode(ref_beauty, ref_mask)
        output = [Xa[0:1].detach().cpu()[:3],ref_beauty[0:1].detach().cpu()[:3]]
        prev_ot_params, ot_params, next_ot_params = self.gen._mlp_output(s_ref)

        prev_prev, _=self.gen.dynamic_mlp.back_activate(prev_ot_params)
        _,next_next = self.gen.dynamic_mlp.back_activate(next_ot_params)

        prev_prev = prev_ot_params+(prev_ot_params-ot_params)
        next_next = next_ot_params+next_ot_params-ot_params

        delta_ = (prev_ot_params-prev_prev ) /3
        for i in range(0,3):
            _params = prev_prev + delta_ * i
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            _image = torch.mul(_image, a_mask) + (-1) * (1 - a_mask)
            _image = _image[0:1].detach().cpu()[:3]
            output.append(_image)

        delta_ = (ot_params-prev_ot_params ) /3
        for i in range(0,3):
            _params = prev_ot_params + delta_ * i
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            _image = torch.mul(_image, a_mask) + (-1) * (1 - a_mask)
            _image = _image[0:1].detach().cpu()[:3]
            output.append(_image)

        delta_ = (next_ot_params-ot_params) /3
        for i in range(0,3):
            _params = ot_params + delta_ * i
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            _image = torch.mul(_image, a_mask) + (-1) * (1 - a_mask)
            _image = _image[0:1].detach().cpu()[:3]
            output.append(_image)

        delta_ = (next_next-next_ot_params ) /3
        for i in range(0,4):
            _params = next_ot_params + delta_ * i
            self.gen.assign_params(_params, self.gen.dec)
            _image = self.gen.dec(c_x)
            _image = torch.mul(_image, a_mask) + (-1) * (1 - a_mask)
            _image = _image[0:1].detach().cpu()[:3]
            output.append(_image)
        final_img=torch.cat(tuple(output), dim=3)
        return final_img







