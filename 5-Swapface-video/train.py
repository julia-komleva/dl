import torch
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from models import Autoencoder
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='FaceSwap-Pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=int, default=1e-4, metavar='N',
                    help='learning_rate')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 100000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-gpu', type=int, default=0, metavar='N',
                    help='number of gpu to train (default: 0)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')




args = parser.parse_args()
print(args)
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(args.num_gpu) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('===> Using GPU to train')
    device = torch.device('cuda:' + str(args.num_gpu))
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
else:
    print('===> Using CPU to train')

torch.manual_seed(args.seed)
# if args.cuda:
    

print('===> Loading datasets')

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.3),
    iaa.Affine(
        scale=(0.95, 1.05),
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-10, 10),
        shear=(-1, 1)
    )
], random_order=True)

def generator(path, num_images=10000, batch = 16, transform = None, seq=None, img_size = (64, 64)):

    pathes = [x.path for x in os.scandir(path) if x.name.endswith(".jpg") 
                 or x.name.endswith(".png") 
                 or x.name.endswith(".bmp")
                 or x.name.endswith(".JPG")]

    np.random.seed(0)
    np.random.shuffle(pathes)
    
    images = np.zeros((len(pathes), 256, 256, 3), dtype=float)
#     input_images = np.zeros((len(pathes), 256, 256, 3), dtype=float)
    for i, pth in enumerate(pathes):
        image = cv2.resize(cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB), (256, 256)) / 255
        images[i] = image
    
    
    
    
#     target_images = rsz(images=target_images)
#     input_images = rsz(images=input_images)
    

    input_images = np.zeros((num_images, *img_size, 3), dtype=float)
    target_images = np.zeros((num_images, *img_size, 3), dtype=float)
    
    wrap = iaa.PiecewiseAffine(scale=0.025, nb_rows=5, nb_cols=5)
    rsz = iaa.Resize({"height": img_size[0], "width": img_size[1]}, interpolation='nearest')
    
    indexes = np.random.randint(0, high=len(images), size=num_images)
    for i, ind in tqdm(enumerate(indexes), total=num_images):
        target_image = images[ind]

        if seq:
            target_image = seq(image=target_image)

        input_image = target_image.copy()

        input_image = wrap(image=input_image)
        target_image = target_image[50:-50, 50:-50, :]
        input_image = input_image[50:-50, 50:-50, :]

        input_images[i] = rsz(image=input_image)
        target_images[i] = rsz(image=target_image)
    while True:
        indexes = np.random.randint(0, high=len(input_images), size=batch)
        yield torch.tensor(input_images[indexes].transpose((0,3,1,2))), torch.tensor(target_images[indexes].transpose((0,3,1,2)))
                
loader_A = generator("train/face_A", seq=seq, batch = args.batch_size)
loader_B = generator("train/face_B", seq=seq, batch = args.batch_size)

model = Autoencoder().to(device)

print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint_' + str(args.num_gpu)):
    try:
        checkpoint = torch.load('./checkpoint_' + str(args.num_gpu) + '/autoencoder.t7')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found autoencoder.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')


criterion = nn.L1Loss()
optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_A.parameters()}]
                         , lr=args.lr, betas=(0.5, 0.999))
optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_B.parameters()}]
                         , lr=args.lr, betas=(0.5, 0.999))

# print all the parameters im model
# s = sum([np.prod(list(p.size())) for p in model.parameters()])
# print('Number of params: %d' % s)

print('Start training, press \'q\' to stop')

for epoch in range(start_epoch, args.epochs):

    input_A, target_A = next(loader_A)
    input_B, target_B = next(loader_B)

    input_A, target_A = input_A.to(device).float(), target_A.to(device).float()
    input_B, target_B = input_B.to(device).float(), target_B.to(device).float()

    optimizer_1.zero_grad()
    optimizer_2.zero_grad()


    res_A = model(input_A, 'A')
    res_B = model(input_B, 'B')

    loss1 = criterion(res_A, target_A)
    loss2 = criterion(res_B ,target_B)
    loss = loss1.item() + loss2.item()
    loss1.backward()
    loss2.backward()
    optimizer_1.step()
    optimizer_2.step()
    print('epoch: {}, lossA:{}, lossB:{}'.format(epoch, loss1.item(), loss2.item()))
    
    if epoch % args.log_interval == 0:

        test_A_ = target_A[0:6]
        test_B_ = target_B[0:6]
        test_A = target_A[0:6].detach().cpu().numpy().transpose((0,2,3,1))
        test_B = target_B[0:6].detach().cpu().numpy().transpose((0,2,3,1))
        print('===> Saving models...')
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint_' + str(args.num_gpu)):
            os.mkdir('checkpoint_' + str(args.num_gpu))
        torch.save(state, './checkpoint_' + str(args.num_gpu) + '/autoencoder.t7')

        figure_A = np.stack([
            test_A,
            model(test_A_, 'A').detach().cpu().numpy().transpose((0,2,3,1)),
            model(test_A_, 'B').detach().cpu().numpy().transpose((0,2,3,1)),
        ], axis=0)
        
        figure_B = np.stack([
            test_B,
            model(test_B_, 'B').detach().cpu().numpy().transpose((0,2,3,1)),
            model(test_B_, 'A').detach().cpu().numpy().transpose((0,2,3,1)),
        ], axis=0)
        
        figure = np.concatenate([figure_A, figure_B], axis=0)

        figure = np.concatenate([it for it in figure],axis=2)

        figure = np.concatenate([it for it in figure],axis=0)

        plt.imsave('train/result_' + str(args.num_gpu) + '.bmp', figure)

#         cv2.imshow("", figure)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             exit()
