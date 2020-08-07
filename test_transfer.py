import torch
from options import TestOptions
from dataset import dataset_single, dataset_single_multi
from model import DRIT
from saver import save_imgs, save_imgs_avg
import os


def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    if opts.multi_modal:
        datasetA = dataset_single_multi(opts, 'A', opts.input_dim_a)
        datasetB = dataset_single_multi(opts, 'B', opts.input_dim_b)
    else:
        datasetA = dataset_single(opts, 'A', opts.input_dim_a)
        datasetB = dataset_single(opts, 'B', opts.input_dim_b)


    if opts.a2b:
        loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads)
        loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads)
        loader_attr = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads, shuffle=True)

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # test
    print('\n--- testing ---')
    for idx1, img1 in enumerate(loader):

        name1 = img1[0][0]
        img1 = img1[1]
        if idx1 == 2000:
            break
        print('{}/{}'.format(idx1 + 1, len(loader)))
        img1 = img1.cuda()
        imgs = [img1]
        names = ['{}'.format(name1)]
        for idx2, img2 in enumerate(loader_attr):
            name2 = img2[0][0]

            img2 = img2[1]
            if idx2 == opts.num:
                break
            img2 = img2.cuda()
            with torch.no_grad():
                if opts.a2b:
                    img, _ = model.test_forward_transfer(img1, img2, a2b=True)
                else:
                    img, _ = model.test_forward_transfer(img2, img1, a2b=False)
            imgs.append(img)
            names.append('{}'.format(name2))

        if opts.average:
            save_imgs_avg(imgs, names, result_dir, opts.multi_modal)
        else:
            save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(name1)), opts.multi_modal)

    return


if __name__ == '__main__':
    main()
