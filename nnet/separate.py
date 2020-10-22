#!/usr/bin/env python3

import os
import time
import argparse

import torch as th
import numpy as np

from conv_tas_net_cd import ConvTasNet

from libs.utils import load_json, get_logger
from libs.audio import write_wav
from libs.dataset import Dataset 

def run(args):
    start = time.time()
    logger = get_logger(
            os.path.join(args.checkpoint, 'separate.log'), file=True)
    
    dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, aux_scp=args.aux_scp)
    
    # Load model
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = ConvTasNet(**nnet_conf)
    cpt_fname = os.path.join(args.checkpoint, "best.pt.tar")
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"]) 
    logger.info("Load checkpoint from {}, epoch {:d}".format(
        cpt_fname, cpt["epoch"]))
    
    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()
    
    with th.no_grad():
        total_cnt = 0
        for i, data in enumerate(dataset):    
            mix = th.tensor(data['mix'], dtype=th.float32, device=device)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device) 
            aux_len = th.tensor(data['aux_len'], dtype=th.float32, device=device)
            key = data['key']
            
            if args.gpuid >= 0:
                mix = mix.cuda()
                aux = aux.cuda()
                aux_len = aux_len.cuda()
                
            # Forward            
            ests = nnet(mix, aux, aux_len)
            ests = ests.cpu().numpy()
            norm = np.linalg.norm(mix.cpu().numpy(), np.inf)
            ests = ests[:mix.shape[-1]]
            
            # for each utts
            logger.info("Separate Utt{:d}".format(total_cnt + 1))
            # norm
            ests = ests * norm / np.max(np.abs(ests))
            write_wav(os.path.join(args.dump_dir, key),
                      ests,
                      fs=args.fs)
            total_cnt += 1   
            break
    
    end = time.time()
    logger.info('Utt={:d} | Time Elapsed: {:.1f}s'.format(total_cnt, end-start))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Separating speech using TD-Speakerbeam')
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Directory of checkpoint")
    parser.add_argument("--gpuid", type=int, default=-1, 
                        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument('--mix_scp', type=str, required=True,
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str, required=True,
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str, required=True,
                        help='aux scp')       
    parser.add_argument('--fs', type=int, default=8000, 
                        help="Sample rate for mixture input")
    parser.add_argument('--dump-dir', type=str, default="sps_tas",
                        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
