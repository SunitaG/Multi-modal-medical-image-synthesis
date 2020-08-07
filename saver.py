import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image


def check_make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# tensor to PIL Image
def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


# save a set of images
def save_imgs(imgs, names, path, multimodal):

    check_make_dir(path)

    for idx, (img, name) in enumerate(zip(imgs, names)):

        # print(idx, name)

        if idx ==0:
            img_name = "Input_" +name
        else:
            img_name = "Output_" +name

        # print(img_name)
        if multimodal:
            flair = img[:, 0:3, :, :]
            t1 = img[:, 3:, :, :]
            save_to_file(tensor2img(flair), path, img_name + '_Flair.png')
            save_to_file(tensor2img(t1), path, img_name + '_T1.png')

        else:
            save_to_file(tensor2img(img), path, img_name + '.png')


def save_imgs_avg(imgs, names, path, multimodal):

    input_path = os.path.join(path, 'Input')
    output_path = os.path.join(path, 'Output')

    check_make_dir(input_path)
    check_make_dir(output_path)

    if multimodal:
        flair = imgs[0][:, 0:3, :, :]
        t1 = imgs[0][:, 3:, :, :]
        save_to_file(tensor2img(flair), input_path, names[0] + '_Flair.png')
        save_to_file(tensor2img(t1), input_path, names[0] + '_T1.png')

        output_list_flair = []
        output_list_t1 = []

        for img, name in zip(imgs[1:], names[1:]):
            output_list_flair.append(tensor2img(img[:, 0:3, :, :]))
            output_list_t1.append(tensor2img(img[:, 3:, :, :]))

        output_img_flair = np.average(output_list_flair, axis=0).astype('uint8')
        output_img_t1 = np.average(output_list_t1, axis=0).astype('uint8')
        save_to_file(output_img_flair, output_path, names[0]+'_Flair.png')
        save_to_file(output_img_t1, output_path, names[0]+'_T1.png')

    else:
        save_to_file(tensor2img(imgs[0]), input_path, names[0] + '.png')

        output_list = []

        for img, name in zip(imgs[1:], names[1:]):
            output_list.append(tensor2img(img))

        output_img = np.average(output_list, axis=0).astype('uint8')
        save_to_file(output_img, output_path, names[0]+'.png')


def save_to_file(img, path, img_name):

    img = Image.fromarray(img)
    img.save(os.path.join(path, img_name))


class Saver():
    def __init__(self, opts):
        self.display_dir = os.path.join(opts.display_dir, opts.name)
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if
                       not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)
        if (total_it + 1) % (self.display_freq * 10) == 0:
            # write img
            image_dis = torchvision.utils.make_grid(model.image_display,
                                                    nrow=model.image_display.size(0) // 2) / 2 + 0.5
            self.writer.add_image('Image', image_dis, total_it)

    # save result images
    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
        elif ep == -1:
            model.save('%s/last.pth' % self.model_dir, ep, total_it)
