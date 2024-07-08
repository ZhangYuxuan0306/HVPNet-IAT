import csv
import os
import argparse
import logging
import sys
sys.path.append("..")
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.bert_model import CNERModel
from processor.dataset import MMPNERProcessor, MMPNERDataset
from modules.train import NERTrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the classes for models and trainers
MODEL_CLASSES = {
    'twitter15': CNERModel,
    'twitter17': CNERModel
}

TRAINER_CLASSES = {
    'twitter15': NERTrainer,
    'twitter17': NERTrainer
}

DATA_PROCESS = {
    'twitter15': (MMPNERProcessor, MMPNERDataset),
    'twitter17': (MMPNERProcessor, MMPNERDataset)
}

# Define the data paths
DATA_PATH = {
    'twitter15': {
        'train': 'data/NER_data/twitter2015/train.txt',
        'dev': 'data/NER_data/twitter2015/valid.txt',
        'test': 'data/NER_data/twitter2015/test.txt',
        'train_auximgs': 'data/NER_data/twitter2015/twitter2015_train_dict.pth',
        'dev_auximgs': 'data/NER_data/twitter2015/twitter2015_val_dict.pth',
        'test_auximgs': 'data/NER_data/twitter2015/twitter2015_test_dict.pth'
    },
    'twitter17': {
        'train': 'data/NER_data/twitter2017/train.txt',
        'dev': 'data/NER_data/twitter2017/valid.txt',
        'test': 'data/NER_data/twitter2017/test.txt',
        'train_auximgs': 'data/NER_data/twitter2017/twitter2017_train_dict.pth',
        'dev_auximgs': 'data/NER_data/twitter2017/twitter2017_val_dict.pth',
        'test_auximgs': 'data/NER_data/twitter2017/twitter2017_test_dict.pth'
    },
}

IMG_PATH = {
    'twitter15': 'data/NER_data/twitter2015_images',
    'twitter17': 'data/NER_data/twitter2017_images',
}

AUX_PATH = {
    'twitter15': {
        'train': 'data/NER_data/twitter2015_aux_images/train/crops',
        'dev': 'data/NER_data/twitter2015_aux_images/val/crops',
        'test': 'data/NER_data/twitter2015_aux_images/test/crops',
    },
    'twitter17': {
        'train': 'data/NER_data/twitter2017_aux_images/train/crops',
        'dev': 'data/NER_data/twitter2017_aux_images/val/crops',
        'test': 'data/NER_data/twitter2017_aux_images/test/crops',
    }
}

# Define the function for setting random seed
def set_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# The main function remains mostly unchanged, except for the removal of MRE-related code

def main():
    # def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_dataset', default='./data/NER_data/twitter15_caption.csv', type=str,
                        help="The name of dataset.")
    parser.add_argument('--dataset_name', default='twitter15', type=str,
    # parser.add_argument('--dataset_name', default='twitter2015', type=str,
                        help="The name of dataset, e.g., 'twitter15' or 'twitter17'.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=15, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.06, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1234, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=4, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='ckpt/NERModel', type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', default=True)
    parser.add_argument('--do_train', default=True)
    parser.add_argument('--only_test', default=True)
    parser.add_argument('--max_seq', default=80, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resources.")

    args = parser.parse_args()
    print("-----------------------------")
    # Assert that the dataset_name is for NER, not for MRE
    # assert args.dataset_name in ['twitter15', 'twitter17'], "dataset_name should be 'twitter15' or 'twitter17'"

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[
        args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed)
    if args.save_path is not None and args.do_train:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    writer = None

    if not args.use_prompt:
        img_path, aux_path = None, None

    caption_data = {}
    with open(args.caption_dataset, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row[next(iter(row))]
            caption_data[key] = row

    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(args, caption_data, processor, transform, img_path, aux_path, args.max_seq,
                                  sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=False)

    dev_dataset = dataset_class(args, caption_data, processor, transform, img_path, aux_path, args.max_seq,
                                mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                pin_memory=False)

    test_dataset = dataset_class(args, caption_data, processor, transform, img_path, aux_path, args.max_seq,
                                 mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 pin_memory=False)

    label_mapping = processor.get_label_mapping()
    label_list = list(label_mapping.keys())
    model = CNERModel(label_list, args)
    model.to(args.device)
    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      label_map=label_mapping, args=args, logger=logger, writer=writer)
    if args.do_train:
        trainer.train()
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        trainer.test()

    torch.cuda.empty_cache()
    #
    # label_mapping = processor.get_label_mapping()
    # label_list = list(label_mapping.keys())
    # model = CNERModel(label_list, args)
    # model.to(args.device)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

