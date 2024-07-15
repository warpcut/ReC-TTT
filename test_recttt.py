
import torch
from dataset import get_data_transforms_test, get_data_transforms
import numpy as np
import random
import os
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d, resnet50_c
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.classifier import Cls, classification_loss
from models.recttt import ReCTTT2
from dataset import CIFAR10_C, CIFAR100_C
import torch.backends.cudnn as cudnn
import argparse
from clean_utils import evaluation_ttt2_semi
import warnings
import copy
import logging
from torchvision.utils import save_image

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO', note=''):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs('{}/weights'.format(save_path), exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log_test_{}.txt'.format(note)))
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


def test(args):
    setup_seed(111)

    batch_size = args.batch_size
    image_size = 32
    crop_size = 32
    data_transform = get_data_transforms(image_size, crop_size)
    data_test_transform = get_data_transforms_test(image_size, crop_size)

    data_root = args.data

    if '100' in args.data:
        num_cls = 100
        train_data, test_data, per = CIFAR100_C(data_root, args.corruption, args.level, data_transform, data_test_transform)
    else: 
        num_cls = 10  
        train_data, test_data, per = CIFAR10_C(data_root, args.corruption, args.level, data_transform, data_test_transform)
    

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)

    if 'wide' in args.save_name:
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=4)
    else:
        encoder, bn = resnet50_c(pretrained=True, custom=True)
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
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(encoder2.parameters()) + \
                                    list(classifier.parameters()) + list(classifier2.parameters()),
                                    lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(list(encoder.parameters()) + list(encoder2.parameters()) + \
                                    list(classifier.parameters()) + list(classifier2.parameters()),
                                    lr=args.lr, weight_decay=1e-5)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(encoder2.parameters()) + \
                                    list(classifier.parameters()) + list(classifier2.parameters()),
                                    lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    checkpoint = torch.load(os.path.join(args.save_dir, args.save_name, 'weights/last.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    ckt_ = copy.deepcopy(checkpoint)
    print_fn('Lr: {}'.format(optimizer.param_groups[0]['lr']))
    alpha = checkpoint['alpha']

    f1_best, acc_best = 0, 0
    best_it = 0
            
    ''' EVALUATION '''

    for i in args.ttt_it:
        test_loss, f1, acc = evaluation_ttt2_semi(model, test_dataloader, optimizer, alpha, device, i, ckt_, args.pseudo_l, args.hm, per)
        
        model.load_state_dict(ckt_['model_state_dict'])
        
        print_fn('Test with {} iter - aux_loss: {:.4f}, F1:{:.3f}, Accuracy:{:.5}'.format(i, test_loss, f1, acc))
        if acc >= acc_best:
            f1_best, acc_best = f1, acc
            best_it = i
   
    ''' TRAIN END '''
    return f1_best, acc_best, best_it


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--note', type=str, default='Recontrast test time training')
    parser.add_argument('--corruption', default='0')
    parser.add_argument('--ttt_it', nargs='+', type=int, default=[0,10])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data', type=str, default='../../nas_data/CIFAR-10-C')
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--hm', action='store_true')
    parser.add_argument('--opt', type=str, default='adamw', choices=['sgd', 'adamw', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pseudo_l', type=str, default='none', choices=['none', 'ce', 'dmt', 'dmt+', 'dmt2', 'flip'])
    parser.add_argument('--save_name', type=str,
                        default='res50_ttt_350')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name), note=args.note)
    print_fn = logger.info
    print_fn(args)

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn('Device: {}'.format(device))
    
    corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    done = ['']

    result_list_best = []

    if args.corruption.isdigit():
        args.corruption = int(args.corruption)
        if args.corruption == 99:
            print_fn('Testing: CIFAR-10.1')
        elif args.corruption == 100:
            print_fn('Testing: CIFAR-10')
        else:
            print_fn('Testing: CIFAR-10C, Testing perturbation: {} - level {}'.format(corruption_types[args.corruption], args.level))
        f1_best, acc_best, best_it = test(args)
        result_list_best.append([corruption_types[args.corruption], f1_best, acc_best, best_it])
        
    else:
        for i, item in enumerate(corruption_types):
            if item in done:
                continue
            print_fn('Testing: CIFAR-10, Testing perturbation: {} - level {}'.format(item, args.level))
            args.corruption = i
            f1_best, acc_best, best_it = test(args)
            result_list_best.append([item, f1_best, acc_best, best_it])

    best_f1 = np.mean([result[-3] for result in result_list_best])
    best_acc = np.mean([result[-2] for result in result_list_best])
    print_fn(result_list_best)
    print_fn('best F1:{:.4f}, best Accuracy:{:.4}'.format(best_f1, best_acc))