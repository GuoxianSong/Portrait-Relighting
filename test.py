
from utils import test_write_images,get_config,prepare_sub_folder
from torch.utils.data import DataLoader
from models import Models
from dataset import My3DDataset
import torch.backends.cudnn as cudnn
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Tester():
    def __init__(self):
        cudnn.benchmark = True
        # Load experiment setting
        config = get_config('configs/OT3.yaml')
        self.trainer = Models(config)
        self.trainer.cuda(config['cuda_device'])

        # Setup logger and output folders

        self.trainer.resume('outputs/OT3/checkpoints', hyperparameters=config,need_opt=False)
        self.trainer.eval()
        self.config = config
        self.dataset = My3DDataset(opts=self.config, is_Train=False)
        self.test_loader = DataLoader(dataset=self.dataset, batch_size=self.config['batch_size']*5, shuffle=False,
                                      num_workers=self.config['nThreads'])
    def eval_all(self):
        xb_rmse, xb_psnr, xb_ssim=0,0,0
        count=0
        with torch.no_grad():
            for it, out_data in enumerate(self.test_loader):
                for j in range(len(out_data)):
                    out_data[j] = out_data[j].cuda(self.config['cuda_device']).detach()
                Xa_out, Yb_out, Xb_out, Xb_mask, Xa_mask, Yb_mask = out_data
                _,_,_,tmp_xb_rmse, tmp_xb_psnr, tmp_xb_ssim = \
                    self.trainer.test_forward(Xa_out, Yb_out, Xb_out, Xa_mask,Yb_mask)

                xb_rmse += tmp_xb_rmse
                xb_psnr += tmp_xb_psnr
                xb_ssim += tmp_xb_ssim
                count+=1
                print(it)

            print('final xb_rmse = %f' %(xb_rmse/count))
            print('final xb_psnr = %f' %(xb_psnr/count))
            print('final xb_ssim = %f' %(xb_ssim/count))


    def test_rotation(self):
        self.dataset.eval_rotation=True
        with torch.no_grad():
            xb_rmse, xb_psnr, xb_ssim = torch.tensor([0.0]*12), torch.tensor([0.0]*12), torch.tensor([0.0]*12)
            count = 0
            for it, out_data in enumerate(self.test_loader):
                for j in range(len(out_data)):
                    if(j==2):
                        for s in range(len(out_data[j])):
                            out_data[j][s] = out_data[j][s].cuda(self.config['cuda_device']).detach()
                    else:
                        out_data[j] = out_data[j].cuda(self.config['cuda_device']).detach()
                Xa_out, Yb_out, rotate, Xa_mask, Yb_mask = out_data
                rmse_,psnr_,ssim_ = self.trainer.test_rotation(Xa_out, Yb_out, rotate, Xa_mask,Yb_mask)

                xb_rmse += rmse_
                xb_psnr += psnr_
                xb_ssim += ssim_
                count += 1
                print(it)
            print(xb_rmse / count)
            print(xb_psnr / count)
            print(xb_ssim / count)







