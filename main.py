import argparse
import os
import datetime
import os.path as osp
import numpy as np
import torch
from pygda.datasets import CitationDataset, AirportDataset, BlogDataset, MAGDataset, TwitchDataset
from torch_geometric.utils import degree
from pygda.models import UDAGCN, A2GNN, GRADE, TDSS
from pygda.models import ASN, SpecReg, GNN
from pygda.models import StruRW, ACDNE, DANE
from pygda.models import AdaGCN, JHGDA, KBL
from pygda.models import DGDA, SAGDA, CWGCN, DGSDA
from pygda.models import DMGNN, PairAlign
from pygda.metrics import eval_micro_f1, eval_macro_f1
from model.ncfm import NCFM
from model.gaa import GAA
from model.hgda import HGDA

#  Citation/ACMv9 DBLPv7 Citationv1
#  Airport/BRAZIL EUROPE USA
#  Blog/Blog1 Blog2
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--device', type=str, default='cuda:3', help='specify cuda devices')
parser.add_argument('--source', type=str, default='ACMv9', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='Citationv1', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in NCFM')
parser.add_argument('--dropout', type=float, default=0.25, help='dropout rate')
parser.add_argument('--s_pnums', type=int, default=0, help='source pseudo numbers')
parser.add_argument('--t_pnums', type=int, default=10, help='target pseudo numbers')
parser.add_argument('--weight', type=float, default=0.5, help='loss weight')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ncfm_epoch', type=int, default=200, help='number of epochs for NCFM')
parser.add_argument('--t_batchsize', type=int, default=2048, help='target batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha parameter for NCFM')
parser.add_argument('--model_type', type=str, default='GAA', help='Specify the model type to use')
args = parser.parse_args()

# Load datasets
if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)

if args.source in {'BRAZIL', 'EUROPE', 'USA'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Airport', args.source)
    source_dataset = AirportDataset(path, args.source)

if args.target in {'BRAZIL', 'EUROPE', 'USA'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Airport', args.target)
    target_dataset = AirportDataset(path, args.target)

if args.source in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Blog', args.source)
    source_dataset = BlogDataset(path, args.source)

if args.target in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Blog', args.target)
    target_dataset = BlogDataset(path, args.target)

if args.source in {'MAG_CN', 'MAG_DE', 'MAG_FR', 'MAG_JP', 'MAG_RU', 'MAG_US'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/MAG', args.source)
    source_dataset = MAGDataset(path, args.source)

if args.target in {'MAG_CN', 'MAG_DE', 'MAG_FR', 'MAG_JP', 'MAG_RU', 'MAG_US'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/MAG', args.target)
    target_dataset = MAGDataset(path, args.target)

if args.source in {'DE', 'EN', 'FR'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Twitch', args.source)
    source_dataset = TwitchDataset(path, args.source)

if args.target in {'DE', 'EN', 'FR'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', './data/Twitch', args.target)
    target_dataset = TwitchDataset(path, args.target)

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)
default_num_features = 241

# Check for x attribute, construct features if missing
if not hasattr(source_data, 'x') or source_data.x is None:
    node_degrees = degree(source_data.edge_index[0], num_nodes=source_data.num_nodes).long()
    source_data.x = torch.nn.functional.one_hot(node_degrees, num_classes=default_num_features).float().to(args.device)

if not hasattr(target_data, 'x') or target_data.x is None:
    node_degrees = degree(target_data.edge_index[0], num_nodes=target_data.num_nodes).long()
    target_data.x = torch.nn.functional.one_hot(node_degrees, num_classes=default_num_features).float().to(args.device)

num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))

# Initialize result lists
micro_f1_scores = []
macro_f1_scores = []

# Train over multiple epochs
for epoch in range(10):
    # Select model based on model_type
    if args.model_type == 'NCFM':
        model = NCFM(
            in_dim=num_features,
            hid_dim=args.nhid,
            num_classes=num_classes,
            device=args.device,
            num_layers=args.num_layers,
            dropout=args.dropout,
            s_pnums=args.s_pnums,
            t_pnums=args.t_pnums,
            weight=args.weight,
            weight_decay=args.weight_decay,
            lr=args.lr,
            epoch=args.ncfm_epoch,
            t_batchsize=args.t_batchsize,
            alpha=args.alpha,
        )
    elif args.model_type == 'TDSS':
        model = TDSS(
            in_dim=num_features,
            hid_dim=args.nhid,
            num_classes=num_classes,
            device=args.device,
            num_layers=args.num_layers,
            dropout=args.dropout,
            s_pnums=args.s_pnums,
            t_pnums=args.t_pnums,
            weight=args.weight,
            weight_decay=args.weight_decay,
            lr=args.lr,
            epoch=args.ncfm_epoch
        )
    elif args.model_type == 'HGDA':
        model = HGDA(
            in_dim=num_features,
            hid_dim=args.nhid,
            num_classes=num_classes,
            device=args.device,
            num_layers=args.num_layers,
            dropout=args.dropout,
            s_pnums=args.s_pnums,
            t_pnums=args.t_pnums,
            weight=args.weight,
            weight_decay=args.weight_decay,
            lr=args.lr,
            epoch=args.ncfm_epoch,
        )
    elif args.model_type == 'GAA':
        model = GAA(
            in_dim=num_features,
            hid_dim=args.nhid,
            num_classes=num_classes,
            device=args.device,
            num_layers=args.num_layers,
            dropout=args.dropout,
            s_pnums=args.s_pnums,
            t_pnums=args.t_pnums,
            weight=args.weight,
            weight_decay=args.weight_decay,
            lr=args.lr,
            epoch=args.ncfm_epoch,
        )
    elif args.model_type == 'DGSDA':
        model = DGSDA(
            in_dim=num_features,
            hid_dim=args.nhid,
            num_classes=num_classes,
            device=args.device,
            num_layers=args.num_layers,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            lr=args.lr,
            epoch=args.ncfm_epoch,
        )
    elif args.model_type == 'GNN':
        model = GNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device, gnn='gcn')
    elif args.model_type == 'A2GNN':
        model = A2GNN(
            in_dim=num_features,
            hid_dim=args.nhid,
            num_classes=num_classes,
            device=args.device,
            num_layers=args.num_layers,
            dropout=args.dropout,
            s_pnums=args.s_pnums,
            t_pnums=args.t_pnums,
            weight=args.weight,
            weight_decay=args.weight_decay,
            lr=args.lr,
            epoch=args.ncfm_epoch,
        )
    elif args.model_type == 'UDAGCN':
        model = UDAGCN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'GRADE':
        model = GRADE(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'ASN':
        model = ASN(in_dim=num_features, hid_dim=args.nhid, hid_dim_vae=args.nhid, num_classes=num_classes,
                    device=args.device)
    elif args.model_type == 'SpecReg':
        model = SpecReg(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device,
                        reg_mode=True)
    elif args.model_type == 'StruRW':
        model = StruRW(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'ACDNE':
        model = ACDNE(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'DANE':
        model = DANE(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device,
                     num_layers=args.num_layers)
    elif args.model_type == 'AdaGCN':
        model = AdaGCN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'JHGDA':
        model = JHGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'KBL':
        model = KBL(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'DGDA':
        model = DGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'SAGDA':
        model = SAGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'CWGCN':
        model = CWGCN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'DMGNN':
        model = DMGNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
    elif args.model_type == 'PairAlign':
        model = PairAlign(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)

    # Train the model
    max_mi, max_ma = model.fit(source_data, target_data)
    print('max_mi:', max_mi)
    print('max_ma:', max_ma)
    micro_f1_scores.append(max_mi)
    macro_f1_scores.append(max_ma)

# Compute mean and std
micro_f1_mean = np.mean(micro_f1_scores)
micro_f1_std = np.std(micro_f1_scores)
macro_f1_mean = np.mean(macro_f1_scores)
macro_f1_std = np.std(macro_f1_scores)
print(micro_f1_scores)
print(f'Mean micro-f1: {micro_f1_mean * 100:.2f} ± {micro_f1_std * 100:.2f}')
print(f'Mean macro-f1: {macro_f1_mean * 100:.2f} ± {macro_f1_std * 100:.2f}')

# 定义一个清理函数
import glob


def cleanup_logs(results_dir, prefix, keep_num=10):
    # 获取所有该迁移场景的 log 文件
    pattern = os.path.join(results_dir, f"{prefix}_*.log")
    files = glob.glob(pattern)

    # 提取 micro-f1 得分
    def extract_score(fname):
        try:
            return float(fname.split('_')[-1][:-4])  # 取最后的数字部分
        except:
            return -float('inf')

    files_sorted = sorted(files, key=extract_score, reverse=True)
    # 需要保留的文件
    keep_files = set(files_sorted[:keep_num])
    # 删除多余的
    for f in files_sorted[keep_num:]:
        os.remove(f)


logfile_name = f"results_t_blogUBE/{args.model_type}_{args.source}_{args.target}_{args.t_batchsize}.log"
with open(logfile_name, 'w') as f:
    f.write("Args:\n")
    for k, v in vars(args).items():
        f.write(f"{k}: {v}\n")
    f.write("\n")
    f.write("Results:\n")
    f.write(f"micro_f1_scores: {micro_f1_scores}\n")
    f.write(f"macro_f1_scores: {macro_f1_scores}\n")
    f.write(f"Mean micro-f1: {micro_f1_mean * 100:.2f} ± {micro_f1_std * 100:.2f}\n")
    f.write(f"Mean macro-f1: {macro_f1_mean * 100:.2f} ± {macro_f1_std * 100:.2f}\n")
# cleanup_logs('results', f"ncfm_{args.source}_{args.target}", keep_num=10)

