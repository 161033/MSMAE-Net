from os.path import isfile, join
from PIL import Image
import numpy as np
import abc
import cv2
import torch.utils.data as data
import random
random.seed(1234567890)
from random import randrange
import imageio
import torch

class BaseData(data.Dataset):

    def __init__(self, args):
        super(BaseData, self).__init__()
        self.crop_size = args.crop_size
        self.file_path = './data/REAL'
        self.file_path_fake = './data/FAKE'
        self.image_names = []
        self.image_class = self._img_list_retrieve()
        for idx, _ in enumerate(self.image_class):
            self.image_names += _

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)

    @abc.abstractmethod
    def _img_list_retrieve():
        pass

    def _resize_func(self, input_img):
        input_img = Image.fromarray(input_img)
        input_img = input_img.resize(self.crop_size, resample=Image.BICUBIC)
        input_img = np.asarray(input_img)
        return input_img

    def get_image(self, image_name, aug_index=None):
        image = imageio.v2.imread(image_name)
        if image.shape[-1] == 4:
            image = self.rgba2rgb(image)
        image = self._resize_func(image)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        return image.permute(2, 0, 1)

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape
        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def data_aug(self, img, data_aug_ind):
        img = Image.fromarray(img)
        if data_aug_ind == 0:
            return np.asarray(img)
        elif data_aug_ind == 1:
            return np.asarray(img.rotate(90, expand=True))
        elif data_aug_ind == 2:
            return np.asarray(img.rotate(180, expand=True))
        elif data_aug_ind == 3:
            return np.asarray(img.rotate(270, expand=True))
        elif data_aug_ind == 4:
            return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
        elif data_aug_ind == 5:
            return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        elif data_aug_ind == 6:
            return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        elif data_aug_ind == 7:
            return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
        else:
            raise Exception('Data augmentation index is not applicable.')
    def generate_4masks(self, mask):

        crop_height, crop_width = self.crop_size
        ma_height, ma_width = mask.shape[:2]
        mask_pil = Image.fromarray(mask)
        if ma_height != crop_height or ma_width != crop_width:
            mask_pil = mask_pil.resize(self.crop_size, resample=Image.BICUBIC)

        (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
        (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
        (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

        mask2 = mask_pil.resize((width2, height2))
        mask3 = mask_pil.resize((width3, height3))
        mask4 = mask_pil.resize((width4, height4))

        mask = np.asarray(mask_pil)
        mask = mask.astype(np.float32) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        mask2 = np.asarray(mask2).astype(np.float32) / 255
        mask2[mask2 > 0.5] = 1
        mask2[mask2 <= 0.5] = 0

        mask3 = np.asarray(mask3).astype(np.float32) / 255
        mask3[mask3 > 0.5] = 1
        mask3[mask3 <= 0.5] = 0

        mask4 = np.asarray(mask4).astype(np.float32) / 255
        mask4[mask4 > 0.5] = 1
        mask4[mask4 <= 0.5] = 0

        mask = torch.from_numpy(mask)
        mask2 = torch.from_numpy(mask2)
        mask3 = torch.from_numpy(mask3)
        mask4 = torch.from_numpy(mask4)

        # print(mask.size(), mask2.size(), mask3.size(), mask4.size())
        return mask, mask2, mask3, mask4

    def get_mask(self, image_name, cls, aug_index=None):
        # authentic
        if cls == 0:
            mask = self.load_mask('', real=True, aug_index=aug_index)
            return_res = [0,0,0,0]
        
        # splice
        elif cls == 1:
            if '.jpg' in image_name:
                mask_name = image_name.replace('fake', 'mask').replace('.jpg', '.png')
            else:
                mask_name = image_name.replace('fake', 'mask').replace('.tif', '.png')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,1,1,1]
        
        # Inpainting
        elif cls == 2:
            mask_name = image_name.replace('/fake/', '/mask/').replace('.png', '.jpg')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,1,1,2]

        # copy-move
        elif cls == 3:
            mask_name = image_name.replace('.png', '.png')
            mask_name = mask_name.replace('CopyMove', 'CopyMove_mask')
            mask = self.load_mask(mask_name, aug_index=aug_index)
            return_res = [1,1,1,3]
        else:
            #print(cls, index)
            raise Exception('class is not defined!')

        return mask, return_res

    def load_mask(self, mask_name, real=False, full_syn=False, gray=True, aug_index=None):
        if real:
            mask = np.zeros(self.crop_size)
        else:
            if not full_syn:
                mask = imageio.v2.imread(mask_name) if not gray else np.asarray(Image.open(mask_name).convert('RGB').convert('L'))
                mask = mask.astype(np.float32)
            else:
                mask = np.ones(self.crop_size)
        mask = self.generate_4masks(mask)
        return mask

    def get_cls(self, image_name):
        if 'authentic' in image_name:
            return_cls = 0
        elif 'splice' in image_name:
            return_cls = 1
        elif 'Inpainting' in image_name:
            return_cls = 2
        elif 'CopyMove' in image_name:
            return_cls = 3
        else:
            print(image_name)
            raise ValueError 
        return return_cls

class TrainData(BaseData):
    def __init__(self, args):
        self.is_train = True
        self.val_num = 150000
        super(TrainData, self).__init__(args)
    def img_retrieve(self, file_text, file_folder, real=True):
        result_list = []
        val_num = self.val_num
        data_path = self.file_path if real else self.file_path_fake
        data_text = join(data_path, file_text)
        data_path = join(data_path, file_folder)
        file_handler = open(data_text)
        contents = file_handler.readlines()
        if self.is_train:
            contents_lst = contents[:val_num]
        else:
            contents_lst = contents[val_num:]
        for content in contents_lst:
            if 'mask' not in content:
                img_name = content.strip()
                img_name = join(data_path, img_name)
                result_list.append(img_name)
        file_handler.close()

        if len(result_list) < val_num:
            mul_factor = (val_num//len(result_list)) + 2
            result_list = result_list * mul_factor
        result_list = result_list[-val_num:]
        return result_list

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.get_cls(image_name)
        aug_index = randrange(0, 8)
        image = self.get_image(image_name, aug_index)
        mask, return_res = self.get_mask(image_name, cls, aug_index)
        return image, mask, return_res[0], return_res[1], return_res[2], return_res[3]

    def _img_list_retrieve(self):
        authentic_names = self.img_retrieve('authentic.txt', 'authentic')
        splice_rank_names     = self.img_retrieve('splice_randmask.txt', 'splice_randmask/fake', False)
        Inpainting_names = self.img_retrieve('Inpainting.txt', 'Inpainting/fake', False)
        copy_move_names   = self.img_retrieve('copy_move.txt', 'CopyMove', False)
        splice_name = self.img_retrieve('splice.txt', 'splice', False)
        return [authentic_names, splice_rank_names, Inpainting_names, copy_move_names, splice_name]

class ValData(BaseData):
    def __init__(self, args):
        self.is_train  = False
        self.val_num   = 1500
        super(ValData, self).__init__(args)
    def img_retrieve(self, file_text, file_folder, real=True):

        result_list = []
        val_num  = self.val_num
        data_path = self.file_path if real else self.file_path_fake
        data_text = join(data_path, file_text)
        data_path = join(data_path, file_folder)
        file_handler = open(data_text)
        contents = file_handler.readlines()
        for content in contents[-val_num:]:
            if 'mask' not in content:
                img_name = content.strip()
                img_name = join(data_path, img_name)
                result_list.append(img_name)
        file_handler.close()

        if len(result_list) < val_num:
            mul_factor  = (val_num//len(result_list)) + 2
            result_list = result_list * mul_factor
        result_list = result_list[-val_num:]
        return result_list
    
    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.get_cls(image_name)
        image = self.get_image(image_name)
        mask, return_res = self.get_mask(image_name, cls)
        return image, mask, return_res[0], return_res[1], return_res[2], return_res[3], image_name

    def _img_list_retrieve(self):
        splice_rank_names = self.img_retrieve('splice_randmask.txt', 'splice_randmask/fake', False)
        splice_name = self.img_retrieve('splice.txt', 'splice', False)
        Inpainting_names = self.img_retrieve('Inpainting.txt', 'Inpainting/fake', False)
        copy_move_names = self.img_retrieve('copy_move.txt', 'CopyMove', False)
        return [splice_rank_names, splice_name, Inpainting_names, copy_move_names]
