
import torch
import numpy as np
import random
import os
import argparse
import torch.backends.cudnn as cudnn
import argparse
import warnings
import copy
import logging

from dataset import get_data_transforms, get_data_transforms_test, get_visda_transforms
from dataset import VISDA
from models.resnet import resnet18, resnet34, resnet50, resnet50_c, wide_resnet50_2, resnext50_32x4d
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.classifier import Cls, classification_loss
from models.recttt import ReCTTT2
from utils import global_cosine, global_cosine_hm, write_res_csv, evaluation_ttt2_semi
from ptflops import get_model_complexity_info
from torchvision.utils import save_image
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.utils.data.distributed

warnings.filterwarnings("ignore")

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs('{}/weights'.format(save_path), exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, device):
    seed = 111
    setup_seed(seed)

    epochs = args.epochs
    batch_size = args.batch_size
    image_size = 256
    crop_size = 224
    per = 'visda'
    data_transform, data_test_transform = get_visda_transforms()

    data_root = args.data

    if args.multi:
        ngpus_per_node = torch.cuda.device_count()
        batch_size = int(args.batch_size / ngpus_per_node)

        print('GPU: {}'.format(ngpus_per_node))
        """ This next line is the key to getting DistributedDataParallel working on SLURM:
            SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
            current process inside a node and is also 0 or 1 in this example."""

        local_rank = int(os.environ.get("SLURM_LOCALID")) 
        rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

        device = local_rank

        torch.cuda.set_device(device)

        """ this block initializes a process group and initiate communications
            between all processes running on all nodes """

        print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
        #init the process group
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)

    print_fn('Device: {}'.format(device))
    train_data, val_split_data, test_data, val_data = VISDA(data_root, data_transform, data_test_transform, seed)
    print('Data loaded')
    num_cls = 12
    if args.multi:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler,
                                                    drop_last=False)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                    drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)

    encoder, bn = resnet50(pretrained=True, custom=False)
    decoder = de_resnet50(pretrained=False, output_conv=4)
    classifier = Cls(num_classes=num_cls)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder2 = copy.deepcopy(encoder)
    encoder_freeze = copy.deepcopy(encoder)
    classifier = classifier.to(device)
    classifier2 = copy.deepcopy(classifier)
    model = ReCTTT2(encoder=encoder, encoder2=encoder2, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder, classifier=classifier, classifier2=classifier2)
    if args.multi:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    ce_loss = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.SGD(list(decoder.parameters()) + list(bn.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(list(encoder.parameters()) + list(encoder2.parameters()) + \
                                list(classifier.parameters()) + list(classifier2.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(25, 50, 75, 100, 125, 150, 175, 200), gamma=0.5)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=(25, 50, 75, 100, 125, 150, 175, 200), gamma=0.5)
    
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))

    f1_best, acc_best = 0, 0
    it = 0
    total_iters = epochs * int((len(train_data)/batch_size))
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    ''' TRAIN '''
    for epoch in range(epochs):
        if args.multi:
            train_sampler.set_epoch(epoch)
        model.train(True)
        
        loss_list = []
        cls_loss_list = []
        aux_loss_list = []
        kl_loss_list = []
        for img, label in tqdm(train_dataloader):
            #save_image(img[0], 'train.png')
            optimizer.zero_grad()
            optimizer2.zero_grad()
            
            img = img.to(device)
            label = label.to(device)
            en, de, cls_p, cls2_p = model(img)
            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            
            aux_loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 4 + \
                       global_cosine_hm(en[3:6], de[3:6], alpha=alpha, factor=0.) / 4 + \
                       global_cosine_hm(en[6:9], de[6:9], alpha=alpha, factor=0.) / 4 + \
                       global_cosine_hm(en[9:], de[9:], alpha=alpha, factor=0.) / 4
            
            loss_div = kl_loss(F.log_softmax(cls_p, 0), F.log_softmax(cls2_p.detach(), 0)) + \
                       kl_loss(F.log_softmax(cls2_p, 0), F.log_softmax(cls_p.detach(), 0))

            loss_cls = ce_loss(cls_p, label) + \
                       ce_loss(cls2_p, label)

            final_loss = aux_loss + loss_cls + loss_div
            
            final_loss.backward()
            optimizer.step()
            optimizer2.step()
        
            loss_list.append(final_loss.item())
            aux_loss_list.append(aux_loss.item())
            cls_loss_list.append(loss_cls.item())
            kl_loss_list.append(loss_div.item())
            it += 1
        
        if args.scheduler:
            scheduler.step()
            scheduler2.step()

        ''' EVALUATION '''
        if epoch+1 == 100:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'loss': final_loss,
                    'alpha' : alpha,
                    'test_loss' : 0,
                    'acc' : 0,
                    },os.path.join(args.save_dir, args.save_name, 'weights/100.pt'))
        if (epoch + 1) % args.eval_ep == 0:

            ## TTT phase
            model_ckt = copy.deepcopy(model.state_dict())
            opt_stdict = copy.deepcopy(optimizer.state_dict())
            opt2_stdict = copy.deepcopy(optimizer2.state_dict())
            ckt_ = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict()
            }
            
            test_loss, f1, acc = evaluation_ttt2_semi(model, test_dataloader, optimizer2, alpha, device, args.ttt_it, ckt_, args.pseudo_l, args.hm, per, args.multi)
        
            model.load_state_dict(model_ckt)
            optimizer.load_state_dict(opt_stdict)
            optimizer2.load_state_dict(opt2_stdict)
            print_fn('epoch [{}/{}], iter:{}, loss:{:.4f}, aux_loss:{:.4f}, cls_loss: {:.4f}, kl_loss: {:.4f} | Test aux_loss: {:.4f}, F1:{:.3f}, Accuracy:{:.3}'.format(epoch, epochs, it, np.mean(loss_list), np.mean(aux_loss_list), np.mean(cls_loss_list), np.mean(kl_loss_list), test_loss, f1, acc))
            
            if acc > acc_best:
                f1_best, acc_best = f1, acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'loss': final_loss,
                    'alpha' : alpha,
                    'test_loss' : test_loss,
                    'acc' : acc,
                    },os.path.join(args.save_dir, args.save_name, 'weights/best.pt'))
            model.train()
        else:
            print_fn('epoch [{}/{}], iter:{}, loss:{:.4f}, aux_loss:{:.4f}, cls_loss: {:.4f}, kl_loss: {:.4f}'.format(epoch, epochs, it, np.mean(loss_list), np.mean(aux_loss_list), np.mean(cls_loss_list), np.mean(kl_loss_list)))
    
    ''' TRAIN END '''
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                'loss': final_loss,
                'alpha' : alpha,
                'test_loss' : test_loss,
                'acc' : acc,
                },os.path.join(args.save_dir, args.save_name, 'weights/last.pt'))
    return np.mean(loss_list), np.mean(aux_loss_list), np.mean(cls_loss_list), test_loss, f1, acc,f1_best, acc_best

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--eval_ep', type=int, default=5)
    parser.add_argument('--data', type=str, default='../../nas_data/VISDA/')
    parser.add_argument('--corruption', default='0')
    parser.add_argument('--ttt_it', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hm', action='store_true')
    parser.add_argument('--pseudo_l', type=str, default='none', choices=['none', 'ce', 'dmt', 'dmt+', 'dmt2', 'flip'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_ttt')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn(args)
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    result_list = []
    result_list_best = []

    if args.corruption.isdigit():
        args.corruption = int(args.corruption)
        if args.corruption == 99:
            print_fn('Testing: CIFAR-10.1')
        elif args.corruption == 100:
            print_fn('Testing: CIFAR-10')
        else:
            print_fn('Training: CIFAR-10, Testing perturbation: {} - level {}'.format(corruption_types[args.corruption], args.level))
        final_loss, aux_loss, cls_loss, test_loss, f1, acc,f1_best, acc_best = train(args, device)
        result_list.append([corruption_types[args.corruption], args.level, args.epochs, args.ttt_it, final_loss, aux_loss, cls_loss, test_loss, f1, acc])
        result_list_best.append([corruption_types[args.corruption], f1_best, acc_best])
        
    else:
        for i, item in enumerate(corruption_types):
            print_fn('Training: CIFAR-10, Testing perturbation: {} - level {}'.format(item, args.level))
            args.corruption = i
            final_loss, aux_loss, cls_loss, test_loss, f1, acc, f1_best, acc_best = train(args, device)
            result_list.append([item, args.level, args.epochs, args.ttt_it, final_loss, aux_loss, cls_loss, test_loss, f1, acc])
            result_list_best.append([item, f1_best, acc_best])

    mean_f1 = np.mean([result[-2] for result in result_list])
    mean_acc = np.mean([result[-1] for result in result_list])
    print_fn(result_list)
    print_fn('mean F1:{:.4f}, mean Accuracy:{:.4}'.format(mean_f1, mean_acc))

    best_f1 = np.mean([result[-2] for result in result_list_best])
    best_acc = np.mean([result[-1] for result in result_list_best])
    print_fn(result_list_best)
    print_fn('best F1:{:.4f}, best Accuracy:{:.4}'.format(best_f1, best_acc))
    write_res_csv(os.path.join(args.save_dir, args.save_name), result_list, result_list_best)
