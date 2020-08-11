"""
network dataset process.
"""
from torch.utils.data.dataset import Dataset
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize,Grayscale
import numpy as np
import torch
import pickle
import os
import cv2

class My3DDataset(Dataset):
    def __init__(self,opts,is_Train=True):
        self.path = opts['data_root']
        self.scene_num = opts['scene_num'] #scale =9
        self.subject_index_num=opts['subject_index_num'] #scale=7
        self.degree_gap = int(opts['degree_gap']/30)



        if(opts['models_name']=='dynamic_human'):
            self.is_use_dynamic = True
        else:
            self.is_use_dynamic = False
        self.is_Train = is_Train


        self.train_list,self.test_list,self.train_subjects,self.train_scenes,self.test_Yb_paths,self.test_Xb_paths \
            = self.split(opts['split_files_path'])
        self.size_train_subjects = len(self.train_subjects)
        self.size_train_scenes = len(self.train_scenes)
        if(self.is_Train):
            self.size = len(self.train_list)
        else:
            self.size = len(self.test_list)
        #transforms = []
        transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        mask_transforms = [Resize((opts['crop_image_height'], opts['crop_image_width']), Image.BICUBIC)]
        mask_transforms.append(ToTensor())
        self.mask_transforms = Compose(mask_transforms)
        self.generateMask()
        self.load_real()


        with open('data/subject_names.txt') as f:
            fr = f.readlines()
        self.subject_full_list = [x.strip() for x in fr]

        self.num_train = len(self.train_list)
        self.one_hot = opts['one_hot']
        self.eval_rotation = False
        self.scale = 2
    #split training/test to ensure no overlap in subject/illumination
    def split(self,split_file_path):
        if os.path.isfile(split_file_path):
            m = pickle.load(open(split_file_path, 'rb'))
            train_list =m['train_list']
            test_list = m['test_list']
            train_subjects=m['train_subject']
            train_scenes=m['train_scenes']
            test_Yb_paths=m['test_Yb_paths']
            test_Xb_paths=m['test_Xb_paths']
            return train_list,test_list,train_subjects,train_scenes,test_Yb_paths,test_Xb_paths
        else:
            paths_ = sorted(glob.glob(self.path + "*"))
            train_list=[]
            test_list=[]
            train_subjects=[]

            with open('data/scene.txt') as f:
                fr = f.readlines()
            outdoor_scene =[x.strip() for x in fr]


            train_scenes =[]
            test_scenes=[]
            #7/3 train/test
            for i in range(len(outdoor_scene)):
                if(i%10==1 or i%10==4 or i%10==8  ):
                    test_scenes.append('Scene'+outdoor_scene[i])
                else:
                    train_scenes.append('Scene'+outdoor_scene[i])
            with open('data/test_files.txt') as f:
                fr = f.readlines()
            patterns =[x.strip() for x in fr]
            for path_  in paths_:
                pattern  = path_.split('/')[-1]
                if(pattern in patterns):
                    #test_list+=glob.glob(path_+'/*/*/*.jpg')
                    #decouple the scene and subject
                    subject_scenes = glob.glob(path_+'/*/*')
                    for _scene in subject_scenes:
                        if(_scene.split('/')[-1] in test_scenes):
                            test_list += glob.glob(_scene + '/*.jpg')
                else:
                    train_subjects.append(pattern)
                    subject_scenes = glob.glob(path_ + '/*/*')
                    for _scene in subject_scenes:
                        if (_scene.split('/')[-1] in train_scenes):
                            train_list += glob.glob(_scene + '/*.jpg')
            test_Yb_paths,test_Xb_paths=self.relight_pairs(test_list)


            split_file = {'train_list': train_list, 'test_list': test_list,'train_subject':train_subjects,
                          'train_scenes':train_scenes,'test_Yb_paths':test_Yb_paths,'test_Xb_paths':test_Xb_paths}
            with open(split_file_path, 'wb') as f:
                pickle.dump(split_file, f)
            return train_list,test_list,train_subjects,train_scenes,test_Yb_paths,test_Xb_paths

    def generateTestDynamic(self,index):
        Xa_path, Xb_path, Ya_path, Yb_path, Xa_prev_path, Xa_next_path, Xb_prev_path, Xb_next_path \
            = self.generateFour(self.test_list[index])
        Xa_out, Xa_mask = self.GetOne(Xa_path)
        Xa_prev_out, _ = self.GetOne(Xa_prev_path)
        Xa_next_out, _ = self.GetOne(Xa_next_path)
        return Xa_out,Xa_prev_out,Xa_next_out,Xa_mask


    def __getitem__(self, index):
        if(self.is_Train):
            Xa_path, Xb_path, Ya_path, Yb_path, Xa_prev_path, Xa_next_path, Xb_prev_path, Xb_next_path \
                =self.generateFour(self.train_list[index])

            Xa_out, Xa_mask = self.GetOne(Xa_path)
            Xb_out, Xb_mask = self.GetOne(Xb_path)
            Yb_out, Yb_mask = self.GetOne(Yb_path)
            rand_y_out, rand_y_mask =self.GetOne(self.train_list[np.random.randint(0, self.num_train)])

            if(self.is_use_dynamic):
                Xb_prev_out, _ = self.GetOne(Xb_prev_path)
                Xb_next_out, _ = self.GetOne(Xb_next_path)
                return Xa_out,Xb_out,Yb_out,Xb_prev_out,Xb_next_out,Xa_mask,Yb_mask,rand_y_out, rand_y_mask
        else:

            if(self.eval_rotation):
                Xa_out, Xa_mask = self.GetOne(self.test_list[index])
                Yb_out, Yb_mask = self.GetOne(self.test_Yb_paths[index])
                #Xb_out, Xb_mask = self.GetOne(self.test_Xb_paths[index])

                tmp = self.test_Xb_paths[index].split('/')
                b_scene = tmp[self.scene_num]
                b_scene_angle = int(tmp[-1].split('.')[1])-1
                X_subject_name = tmp[self.subject_index_num]
                rotate=[]
                for i in range(12):
                    _angle = '{:02}'.format((b_scene_angle-6+i) % 12 + 1)
                    _path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + _angle + '.jpg'
                    rotate.append(self.transforms(Image.open(_path).convert('RGB')))
                return Xa_out,Yb_out,rotate, Xa_mask,Yb_mask

            Xa_out, Xa_mask= self.GetOne(self.test_list[index])
            Yb_out, Yb_mask = self.GetOne(self.test_Yb_paths[index])
            Xb_out, Xb_mask = self.GetOne(self.test_Xb_paths[index])
            return Xa_out,Yb_out,Xb_out, Xb_mask,Xa_mask,Yb_mask



    def GetOne(self,path_ ):
        beauty= self.transforms(Image.open(path_).convert('RGB'))
        pattern = path_.split('/')[-1].split('.')[0]
        mask_ =self.mask_dir[pattern]
        #beauty = torch.mul(beauty,mask_)
        return beauty,mask_



    # Xa,Yb, Xb, Ya
    def generateFour(self,Xa_path):
        tmp = Xa_path.split('/')
        a_scene_angle  =tmp[-1].split('.')[1]
        a_scene = tmp[self.scene_num]
        X_subject_name = tmp[self.subject_index_num]

        #random subject + scene
        b_scene = tmp[self.scene_num]
        b_scene_angle = tmp[-1].split('.')[1]
        Y_subject_name = X_subject_name
        while (b_scene == a_scene):
            b_scene = self.train_scenes[np.random.randint(0, self.size_train_scenes)]
            b_scene_angle = '{:02}'.format(np.random.randint(1, 13))
        while(Y_subject_name==X_subject_name):
            Y_subject_name = self.train_subjects[np.random.randint(1, self.size_train_subjects)]
        Yb_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_scene_angle + '.jpg'
        Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_scene_angle + '.jpg'
        Ya_path = self.path + Y_subject_name + '/data/' + a_scene + '/' + Y_subject_name + '.' + a_scene_angle + '.jpg'

        if(self.one_hot==True):
            # dynamic
            a_prev_angle, a_next_angle = self.dynamic_angle(a_scene_angle)
            b_prev_angle, b_next_angle = self.dynamic_angle(b_scene_angle)

            Yb_prev_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_prev_angle + '.jpg'
            Yb_next_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_next_angle + '.jpg'

            Xb_prev_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_prev_angle + '.jpg'
            Xb_next_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_next_angle + '.jpg'

            Yb_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_scene_angle + '.jpg'
            Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_scene_angle + '.jpg'
            Ya_path = self.path + Y_subject_name + '/data/' + a_scene + '/' + Y_subject_name + '.' + a_scene_angle + '.jpg'
            return Xa_path, Xb_path, Ya_path, Yb_path, Yb_prev_path, Yb_next_path, Xb_prev_path, Xb_next_path
        elif(self.is_use_dynamic):
            # dynamic
            a_prev_angle, a_next_angle = self.dynamic_angle(a_scene_angle)
            b_prev_angle, b_next_angle = self.dynamic_angle(b_scene_angle)

            Xa_prev_path = self.path + X_subject_name + '/data/' + a_scene + '/' + X_subject_name + '.' + a_prev_angle + '.jpg'
            Xa_next_path = self.path + X_subject_name + '/data/' + a_scene + '/' + X_subject_name + '.' + a_next_angle + '.jpg'

            Xb_prev_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_prev_angle + '.jpg'
            Xb_next_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_next_angle + '.jpg'

            Yb_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_scene_angle + '.jpg'
            Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_scene_angle + '.jpg'
            Ya_path = self.path + Y_subject_name + '/data/' + a_scene + '/' + Y_subject_name + '.' + a_scene_angle + '.jpg'
            return Xa_path, Xb_path, Ya_path, Yb_path, Xa_prev_path, Xa_next_path, Xb_prev_path, Xb_next_path
        else:
            return Xa_path,Xb_path,Ya_path,Yb_path


    def load_real(self):
        f = open('data/real_image.txt')
        fl = f.readlines()
        real_files =[]
        for i in range(len(fl)):
            real_files.append(fl[i].split(',')[0])
        self.real_files=real_files

    def getReal(self,index):
        file_path = self.path+'../Real_Image/'+self.real_files[index]
        beauty = self.transforms(Image.open(file_path+'.jpg').convert('RGB'))
        mask_ = self.mask_transforms(Image.open(file_path+'.tif').convert('RGB'))
        return beauty, mask_


    def dynamic_angle(self,angle):
        angle = int(angle)-1
        prev_angle = '{:02}'.format((angle-self.degree_gap)%12+1)
        next_angle = '{:02}'.format((angle+self.degree_gap)%12+1)
        return prev_angle,next_angle

    def generateMask(self):
        mask_path = self.path+'..'+'/Mask/*/data/albedo/*.png'
        dirs = glob.glob(mask_path)
        mask_dir={}
        for i in range(len(dirs)):
            pattern = dirs[i].split('/')[-1].split('.')[0]
            img = self.mask_transforms(Image.open(dirs[i]).convert('RGB'))
            mask_dir[pattern]=img
        self.mask_dir=mask_dir

    #inpaint Yb background in test dataset
    def inpaint(self,index,path=None,mask_path =None):
        if(path is None):
            Yb_img = cv2.imread(self.test_Yb_paths[index])
            pattern = self.test_Yb_paths[index].split('/')[-1].split('.')[0]
        else:
            Yb_img = cv2.imread(path)
            pattern = path.split('/')[-1].split('.')[0]
        if(mask_path is None):
            mask_path = self.path + '..' + '/Mask/'+pattern+'/data/albedo/'+pattern+'.png'
        Yb_mask = cv2.imread(mask_path,0)
        Yb_mask[Yb_mask != 0] = 255
        kernel = np.ones((10, 10), np.uint8)
        Yb_mask = cv2.dilate(Yb_mask, kernel, iterations=2)
        dst = cv2.inpaint(Yb_img, Yb_mask, 3, cv2.INPAINT_TELEA)

        img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)


        transforms = [Resize((256*self.scale, 256*self.scale), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        beatuy_transforms = Compose(transforms)

        img = beatuy_transforms(im_pil)
        return img

    def relight_pairs(self,test_list):

        Yb_paths=[]
        Xb_paths=[]
        full_subjects_num = len(self.subject_full_list)
        for i in range(len(test_list)):
            Xa_path = test_list[i]
            tmp = Xa_path.split('/')
            a_scene_angle = tmp[-1].split('.')[1]
            a_scene = tmp[self.scene_num]
            X_subject_name = tmp[self.subject_index_num]

            # random subject + scene
            b_scene = tmp[self.scene_num]
            b_scene_angle = tmp[-1].split('.')[1]
            Y_subject_name = X_subject_name
            while (b_scene == a_scene):
                b_scene = 'Scene'+str(np.random.randint(1, 323))
                b_scene_angle = '{:02}'.format(np.random.randint(1, 13))

            while (Y_subject_name == X_subject_name):
                Y_subject_name = self.subject_full_list[np.random.randint(0, full_subjects_num)]

            Yb_path = self.path + Y_subject_name + '/data/' + b_scene + '/' + Y_subject_name + '.' + b_scene_angle + '.jpg'
            Xb_path = self.path + X_subject_name + '/data/' + b_scene + '/' + X_subject_name + '.' + b_scene_angle + '.jpg'
            Yb_paths.append(Yb_path)
            Xb_paths.append(Xb_path)
        return Yb_paths,Xb_paths


    def __len__(self):
        return self.size


    def getReal(self,index):
        file_path = self.path+'../Real_Image/'+self.real_files[index]
        beauty = self.transforms(Image.open(file_path+'.jpg').convert('RGB'))
        mask_ = self.mask_transforms(Image.open(file_path+'.tif').convert('RGB'))
        return beauty, mask_

    def getReal2(self,index):
        transforms = [Resize((256*self.scale, 256*self.scale), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        beatuy_transforms = Compose(transforms)

        mask_transforms = [Resize((256*self.scale, 256*self.scale), Image.BICUBIC)]
        mask_transforms.append(ToTensor())
        mask_transforms = Compose(mask_transforms)


        file_path = self.path+'../Real_Image/'+self.real_files[index]
        beauty = beatuy_transforms(Image.open(file_path+'.jpg').convert('RGB'))
        mask_ = mask_transforms(Image.open(file_path+'.tif').convert('RGB'))
        return beauty, mask_