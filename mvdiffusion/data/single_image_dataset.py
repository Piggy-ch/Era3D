from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math
import rembg

from glob import glob

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb
from icecream import ic

import cv2
import numpy as np

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def scale_and_place_object(image, scale_factor):
    assert np.shape(image)[-1]==4  # RGBA

    # Extract the alpha channel (transparency) and the object (RGB channels)
    alpha_channel = image[:, :, 3]

    # Find the bounding box coordinates of the object
    coords = cv2.findNonZero(alpha_channel)
    x, y, width, height = cv2.boundingRect(coords)

    # Calculate the scale factor for resizing
    original_height, original_width = image.shape[:2]

    if width > height:
        size = width
        original_size = original_width
    else:
        size = height
        original_size = original_height

    scale_factor = min(scale_factor, size / (original_size+0.0))

    new_size = scale_factor * original_size
    scale_factor = new_size / size

    # Calculate the new size based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    center_x = original_width // 2
    center_y = original_height // 2

    paste_x = center_x - (new_width // 2)
    paste_y = center_y - (new_height // 2)

    # Resize the object (RGB channels) to the new size
    rescaled_object = cv2.resize(image[y:y+height, x:x+width], (new_width, new_height))

    # Create a new RGBA image with the resized image
    new_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)

    new_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = rescaled_object

    return new_image

class SingleImageDataset(Dataset):
    def __init__(self,
        root_dir: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        cond_images: Optional[list] = None,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.bg_remover = rembg.new_session()

        
        if single_image is None:
            if filepaths is None:
                # Get a list of all files in the directory
                file_list = os.listdir(self.root_dir)
            else:
                file_list = filepaths

            # Filter the files that end with .png or .jpg
            self.file_list = [file for file in file_list if file.endswith(('.png', '.jpg', '.webp'))]
        else:
            self.file_list = None

        # load all images
        self.all_images = []
        self.all_cond_images = []
        self.all_cond_normals = []
        self.all_alphas = []
        bg_color = self.get_bg_color()

        if single_image is not None:
            image, alpha = self.load_image(None, bg_color, return_type='pt', Imagefile=single_image)
            self.all_images.append(image)
            self.all_alphas.append(alpha)
        else:
            for file in self.file_list:
                print(os.path.join(self.root_dir, file))
                image, alpha = self.load_image(os.path.join(self.root_dir, file), bg_color, return_type='pt')
                cond_images,cond_normals = self.load_cond_image(os.path.join("..","val_test", '0'), bg_color, return_type='pt')
                self.all_images.append(image)
                self.all_alphas.append(alpha)
                self.all_cond_images.append(cond_images)
                self.all_cond_normals.append(cond_normals)
                
            

        self.all_images = self.all_images[:num_validation_samples]
        self.all_alphas = self.all_alphas[:num_validation_samples]
        self.all_cond_images = self.all_cond_images[:num_validation_samples]
        self.all_cond_normals = self.all_cond_normals[:num_validation_samples]
        ic(len(self.all_images))
        
        try:
            self.normal_text_embeds = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/clr_embeds.pt') # 4view
        except:
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/embeds.pt')
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.all_images)

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    
    def load_image(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]
        if image_input.mode != 'RGBA':
            image_input = rembg.remove(image_input, session=self.bg_remover)
        if self.crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        
        return img, alpha
    
    def load_cond_image(self, img_path, bg_color, return_type='np', Imagefile=None):
        image_size = self.img_wh[0]
        rgb_list = []
        normal_list = []
        for i in range(6):
            rgb_file = img_path+'_rgb_or'+str(i)+'.jpg'
            normal_file = img_path+'_normal_or'+str(i)+'.jpg'
            alpha_file = img_path+'_alpha_or'+str(i)+'.jpg'
            rgb_i = Image.open(rgb_file)
            normal_i = Image.open(normal_file)
            alpha_i = Image.open(alpha_file)
            # 将alpha图像对象转换为NumPy数组  
            alpha_array = np.array(alpha_i)
            # 对数组进行操作  
            alpha_array[alpha_array < 0.1] = 0  
            # 如果需要，你可以将修改后的数组转换回图像对象  
            alpha_i = Image.fromarray(alpha_array)  
            rgb_i = rgb_i.resize((image_size, image_size))
            normal_i = normal_i.resize((image_size, image_size))
            alpha_i = alpha_i.resize((image_size, image_size))
            # rgb_i.save('t0.png')
            # normal_i.save('t1.png')
            # if self.crop_size != -1:
            #     coords = np.stack(np.nonzero(alpha_i), 1)[:,(1, 0)]
            #     min_x, min_y = np.min(coords, 0)
            #     max_x, max_y = np.max(coords, 0)
            #     ref_img_ = rgb_i.crop((min_x, min_y, max_x, max_y))
            #     ref_normal_ = normal_i.crop((min_x, min_y, max_x, max_y))
            #     h, w = ref_img_.height, ref_img_.width
            #     scale = self.crop_size / max(h, w)
            #     h_, w_ = int(scale * h), int(scale * w)
            #     ref_img_ = ref_img_.resize((w_, h_))
            #     ref_normal_ = ref_normal_.resize((w_, h_))
            #     rgb_i = add_margin(ref_img_, color=1,size=image_size)
            #     normal_i = add_margin(ref_normal_, color=1,size=image_size)
            #     rgb_i.save('t0.png')
            #     normal_i.save('t1.png')
            # else:
            #     rgb_i = add_margin(rgb_i, size=max(rgb_i.height, rgb_i.width))
            #     rgb_i = rgb_i.resize((image_size, image_size))
            #     normal_i = add_margin(normal_i, size=max(normal_i.height, normal_i.width))
            #     normal_i = normal_i.resize((image_size, image_size))
                
            rgb_i = np.array(rgb_i)
            normal_i = np.array(normal_i)
            rgb_i= rgb_i.astype(np.float32) / 255
            normal_i = normal_i.astype(np.float32) / 255
            if return_type == "np":
                pass
            elif return_type == "pt":
                rgb_i = torch.from_numpy(rgb_i)
                normal_i = torch.from_numpy(normal_i)
            else:
                raise NotImplementedError
            rgb_list.append(rgb_i)
            normal_list.append(normal_i)
        if return_type == "np":
            rgb_list = np.stack(rgb_list, axis=0)
            normal_list = np.stack(normal_list, axis=0)
        elif return_type == "pt":
            rgb_list = torch.stack(rgb_list, dim=0)
            normal_list = torch.stack(normal_list, dim=0)
        else:
            raise NotImplementedError
        return rgb_list, normal_list




    def __getitem__(self, index):
        image = self.all_images[index%len(self.all_images)]
        alpha = self.all_alphas[index%len(self.all_images)]
        cond_images = self.all_cond_images[index%len(self.all_images)]
        cond_normals = self.all_cond_normals[index%len(self.all_images)]
        if self.file_list is not None:
            filename = self.file_list[index%len(self.all_images)].replace(".png", "")
        else:
            filename = 'null'
        img_tensors_in = [
            image.permute(2, 0, 1)
        ] * self.num_views

        alpha_tensors_in = [
            alpha.permute(2, 0, 1)
        ] * self.num_views

        cond_images_in = cond_images.permute(0, 3, 1, 2)
        cond_normals_in = cond_normals.permute(0, 3, 1, 2)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        alpha_tensors_in = torch.stack(alpha_tensors_in, dim=0).float() # (Nv, 3, H, W)
        
        if self.gt_path is not None:
            gt_image = self.gt_images[index%len(self.all_images)]
            gt_alpha = self.gt_alpha[index%len(self.all_images)]
            gt_img_tensors_in = [gt_image.permute(2, 0, 1) ] * self.num_views
            gt_alpha_tensors_in = [gt_alpha.permute(2, 0, 1) ] * self.num_views
            gt_img_tensors_in = torch.stack(gt_img_tensors_in, dim=0).float()
            gt_alpha_tensors_in = torch.stack(gt_alpha_tensors_in, dim=0).float()
                
        normal_prompt_embeddings = self.normal_text_embeds if hasattr(self, 'normal_text_embeds') else None
        color_prompt_embeddings = self.color_text_embeds if hasattr(self, 'color_text_embeds') else None
        
        out =  {
            'imgs_in': img_tensors_in,
            'cond_images': cond_images_in,
            'cond_normals': cond_normals_in,
            'alphas': alpha_tensors_in,
            'normal_prompt_embeddings': normal_prompt_embeddings,
            'color_prompt_embeddings': color_prompt_embeddings,
            'filename': filename,
            }
            
        return out

        


class SinImageDataset(Dataset):
    def __init__(self,
        root_dir: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.bg_remover = rembg.new_session()
        
        
        images_path = os.path.join(root_dir, 'images')
        dirs = os.listdir(images_path)
        self.file_list = [d for d in dirs if os.path.isdir(os.path.join(images_path, d))]

        self.already_path = os.path.join(root_dir, 'already.txt')
        with open(self.already_path, 'r') as f:
            already_list = f.read().splitlines()
        
        # 从file_list中移除already.txt中存在的项
        self.file_list = [d for d in self.file_list if d not in already_list]


        # # load all images
        # self.all_images = []
        # self.all_cond_images = []
        # self.all_cond_normals = []
        # self.all_alphas = []
        # bg_color = self.get_bg_color()

        # if single_image is not None:
        #     image, alpha = self.load_image(None, bg_color, return_type='pt', Imagefile=single_image)
        #     self.all_images.append(image)
        #     self.all_alphas.append(alpha)
        # else:
        #     for file in self.file_list:
        #         print(os.path.join(self.root_dir,'images', file))
        #         image, alpha = self.load_image(os.path.join(self.root_dir, 'images', file,'00.png'), bg_color, return_type='pt')
        #         self.all_images.append(image)
        #         self.all_alphas.append(alpha)
                
            

        # self.all_images = self.all_images[:num_validation_samples]
        # self.all_alphas = self.all_alphas[:num_validation_samples]
        # ic(len(self.all_images))
        
        try:
            self.normal_text_embeds = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/clr_embeds.pt') # 4view
        except:
            self.color_text_embeds = torch.load(f'{prompt_embeds_path}/embeds.pt')
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    
    def load_image(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]
        if image_input.mode != 'RGBA':
            image_input = rembg.remove(image_input, session=self.bg_remover)
        image_input = image_input.resize((image_size, image_size))
        if self.crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError
        return img, alpha
        




    def __getitem__(self, index):
        images_path = self.file_list[index%len(self.file_list)]
        bg_color = self.get_bg_color()
        image, alpha = self.load_image(os.path.join(self.root_dir, 'images', images_path,'00.png'), bg_color, return_type='pt')

        if self.file_list is not None:
            filename = self.file_list[index%len(self.file_list)].replace(".png", "")
        else:
            filename = 'null'
        img_tensors_in = [
            image.permute(2, 0, 1)
        ] * self.num_views

        alpha_tensors_in = [
            alpha.permute(2, 0, 1)
        ] * self.num_views

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        alpha_tensors_in = torch.stack(alpha_tensors_in, dim=0).float() # (Nv, 3, H, W)
        
        if self.gt_path is not None:
            gt_image = self.gt_images[index%len(self.all_images)]
            gt_alpha = self.gt_alpha[index%len(self.all_images)]
            gt_img_tensors_in = [gt_image.permute(2, 0, 1) ] * self.num_views
            gt_alpha_tensors_in = [gt_alpha.permute(2, 0, 1) ] * self.num_views
            gt_img_tensors_in = torch.stack(gt_img_tensors_in, dim=0).float()
            gt_alpha_tensors_in = torch.stack(gt_alpha_tensors_in, dim=0).float()
                
        normal_prompt_embeddings = self.normal_text_embeds if hasattr(self, 'normal_text_embeds') else None
        color_prompt_embeddings = self.color_text_embeds if hasattr(self, 'color_text_embeds') else None
        
        out =  {
            'imgs_in': img_tensors_in,
            'alphas': alpha_tensors_in,
            'normal_prompt_embeddings': normal_prompt_embeddings,
            'color_prompt_embeddings': color_prompt_embeddings,
            'filename': filename,
            }
            
        return out