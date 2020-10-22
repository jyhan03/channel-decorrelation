#!/usr/bin/env python3

import os
import time 
import argparse

import torch as th
import numpy as np
from mir_eval.separation import bss_eval_sources

from conv_tas_net_cd import ConvTasNet

from libs.utils import load_json, get_logger 
from libs.dataset import Dataset 


def evaluate(args):
    start = time.time() 
    total_SISNR = 0
    total_SDR = 0
    total_cnt = 0

    # build the logger object
    logger = get_logger(
            os.path.join(args.checkpoint, 'eval.log'), file=True)
    
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
  
    # Load data
    dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, aux_scp=args.aux_scp)
    
    with th.no_grad():
        for i, data in enumerate(dataset):    
            mix1 = th.tensor(data['mix1'], dtype=th.float32, device=device)
            mix2 = th.tensor(data['mix2'], dtype=th.float32, device=device)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device) 
            aux_len = th.tensor(data['aux_len'], dtype=th.float32, device=device)
            
            if args.gpuid >= 0:
                mix1 = mix1.cuda()
                mix2 = mix2.cuda()
                aux = aux.cuda()
                aux_len = aux_len.cuda()
                
            # Forward
            ref = data['ref']                
            ests, _ = nnet(mix1, mix2,  aux, aux_len)
            ests = ests.cpu().numpy()
            if ests.size != ref.size:
                end = min(ests.size, ref.size)
                ests = ests[:end]
                ref = ref[:end]
                        
            # for each utts
            # Compute SDRi
            if args.cal_sdr:
                SDR, sir, sar, popt = bss_eval_sources(ref, ests)
                # avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDR += SDR[0]
            # Compute SI-SNR
            SISNR = cal_SISNR(ests, ref)
            if args.cal_sdr:
                logger.info("Utt={:d} | SDR={:.2f} | SI-SNR={:.2f}".format(total_cnt+1, SDR[0], SISNR))
            else:
                logger.info("Utt={:d} | SI-SNR={:.2f}".format(total_cnt+1, SISNR))
            total_SISNR += SISNR
            total_cnt += 1  
    end = time.time()
    
    logger.info('Time Elapsed: {:.1f}s'.format(end-start))
    if args.cal_sdr:
        logger.info("Average SDR: {0:.2f}".format(total_SDR / total_cnt))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
                    

def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi        

def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi    

def cal_SISNR(est, ref, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm)**2 + eps)
        
    return 20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))
                         
                         
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model file created by training')
    parser.add_argument('--gpuid', type=int, default=-1,
                        help="GPU device to offload model to, -1 means running on CPU")  
    parser.add_argument('--mix_scp', type=str, required=True,
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str, required=True,
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str, required=True,
                        help='aux scp')    
    
    parser.add_argument('--cal_sdr', type=int, default=0,
                        help='Whether calculate SDR, add this option because calculation of SDR is very slow')

    args = parser.parse_args()
    print(args)
    evaluate(args)
    
