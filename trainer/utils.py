import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2


def gen_input_mask(
        shape, hole_size, hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for _ in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
    return mask


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input.shape[0]
    #print(num_samples)
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i]) #mode=RGB size=128*128
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])#mode=RGB size=128*128
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]] #3*128*128
        out = transforms.functional.to_tensor(out) #[3,128,128]
        out = torch.unsqueeze(out, dim=0)#[1,3,128,128]
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret
    
def vis_parsing_maps(im, parsing_anno,x,y, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    #print("xingzhuang",vis_im.shape)
    
    
    vis_parsing_anno = parsing_anno.copy()
    #print("ano_xingzhuang",vis_parsing_anno.shape)
    
    vis_parsing_anno_color = np.zeros((im.shape[0], im.shape[1], 3)) + 0
    #print("color_xingzhuang",vis_parsing_anno_color.shape)

    face_mask = np.zeros((im.shape[0], im.shape[1]))
    #print("face_maskxingzhuang",face_mask.shape)
    #print("weidu",face_mask.ndim)
    
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)# 

        idx_y = (index[0]+y).astype(np.int)
        idx_x = (index[1]+x).astype(np.int)

        # continue
        vis_parsing_anno_color[idx_y,idx_x, :] = part_colors[pi]# 
        face_mask[idx_y,idx_x] = 0.45
        # if pi in[1,2,3,4,5,6,7,8,10,11,12,13,14,17]:
        #     face_mask[idx_y,idx_x] = 0.35

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    
    #print("color_xingzhuang",vis_parsing_anno_color.shape)
    
    face_mask = np.expand_dims(face_mask, 2)
    #print("xingzhuang",face_mask.shape)
    
    vis_im = vis_parsing_anno_color*face_mask + (1.-face_mask)*vis_im
    vis_im = vis_im.astype(np.uint8)
    #print(vis_im)
    return vis_im
    
def vis_parsing(img,parsing):
    img = img.clone().cpu()
    parsing = parsing.clone().cpu()
    num_samples1 = img.shape[0]
    ret1 = []
    for i in range(num_samples1):
        dst = transforms.functional.to_pil_image(img[i]) #mode=RGB size=128*128
        dst= np.array(dst)[:, :, [2, 1, 0]]
        #src= transforms.functional.to_pil_image(parsing[i])#mode=RGB size=128*128
        #src = np.array(src)[:, :, [2, 1, 0]]
        parsing_ = parsing[i].squeeze(0).cpu().numpy().argmax(0)
        parsing_ = parsing_.astype(np.uint8)
        
        vis_im = vis_parsing_maps(dst, parsing_, 0,0,stride=1)#numpy
        vis_im = vis_im[:, :, [2, 1, 0]] #3*128*128
        vis_im = transforms.functional.to_tensor(vis_im) #[3,128,128]
        vis_im = torch.unsqueeze(vis_im, dim=0)#[1,3,128,128]
        ret1.append(vis_im)
    ret1 = torch.cat(ret1, dim=0)
    return ret1
    
def tensor_argmax(out):
    out = out.clone().cpu()
    num_samples1 = out.shape[0]
    ret1 = []
    for i in range(num_samples1):
        out_ = out[i].squeeze(0).cpu().numpy().argmax(0)
        #out_ = out_.astype(np.uint8)
        out_ = transforms.functional.to_tensor(out_) #[3,128,128]
        out_ = torch.unsqueeze(out_, dim=0)#[1,3,128,128] 
        ret1.append(out_)
    ret1 = torch.cat(ret1, dim=0)
    return ret1
    
    