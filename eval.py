"""This file contains code for evaluation.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/baofff/U-ViT/blob/main/eval.py
"""

import ml_collections
import accelerate
import utils
import tempfile
import builtins
import math

from absl import logging

import torch
from torch import multiprocessing as mp

import libs.autoencoder
from datasets import get_dataset
from dpm_solver import NoiseScheduleVP, DPM_Solver
from tools.fid_score import calculate_fid_given_paths


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    if config.cfg==-1:
        cfg_scale = config.sample.scale
        cfg_scale_pow = config.cfg_scale_pow
    else:
        cfg_scale = config.cfg
        cfg_scale_pow = config.cfg_scale_pow
    if 'cfg' in config.sample and config.sample.cfg and cfg_scale > 0 and cfg_scale_pow:  # classifier free guidance
        logging.info(f'Use classifier free guidance with scale={cfg_scale}, cfg_scale_pow={cfg_scale_pow}')
        def cfg_nnet(x, timesteps, log_snr, y):
            # raise NotImplementedError
            _cond = nnet(x, log_snr=log_snr, y=y)[-1]
            # import pdb; pdb.set_trace()
            _uncond = nnet(x, log_snr=log_snr, y=torch.tensor([dataset.K] * x.size(0), device=device))[-1]

            if cfg_scale_pow == -1:
                # no cfg decay:
                assert cfg_scale < 1
                return _cond + cfg_scale * (_cond - _uncond)
            else:
                # dit cfg 
                assert cfg_scale_pow > 0
                assert cfg_scale >= 1

                # cfg decay
                diffusion_steps = 1000
                assert torch.all(timesteps == timesteps[0])
                scale_step = (1-torch.cos(((1-timesteps[0]/diffusion_steps)**cfg_scale_pow)*math.pi))*1/2
                cfg_scale_decay = (cfg_scale-1)*scale_step + 1

                assert cfg_scale_decay >= 1
                return _uncond + cfg_scale_decay * (_cond - _uncond)
    else:
        def cfg_nnet(x, timesteps, log_snr, y):
            _cond = nnet(x, log_snr=log_snr, y=y)[-1]
            return _cond

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    def sample_z(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)

        if config.sample.algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
            
            def model_fn(x, t_continuous, scaling_factor=-1):
                t = t_continuous * N
                log_snr = (t.float() / 1000) * 8 - 4
                if scaling_factor!=-1:
                    print("using scaling factor to increase noise scheldule for 512 x 512")
                    x = x / x.std(axis=(1,2,3), keepdims=True) 

                eps_pre = cfg_nnet(x, t, log_snr, **kwargs)
                return eps_pre

            if 'scaling_factor' in config:
                scaling_factor = config.scaling_factor
            else:
                scaling_factor = -1
            
            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False, scaling_factor=scaling_factor)
            _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)

        else:
            raise NotImplementedError

        return _z

    def sample_fn(_n_samples, category=-1, accelerator=None):
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, category, accelerator, device=device))
        else:
            raise NotImplementedError
        _z = sample_z(_n_samples, _sample_steps=config.sample.sample_steps, **kwargs)
        return decode_large_batch(_z)

    with tempfile.TemporaryDirectory() as temp_path:
        # path = config.sample.path or temp_path
        path = config.IMGsave_path or config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, evenly_sample=True)
        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'nnet_path={config.nnet_path}, fid={fid}')
            with open("results_searching_cfg.txt", "a") as file:
                file.write(f'FID50K: nnet_path={config.nnet_path} \n ----> cfg={cfg_scale}, cfg_scale_pow={cfg_scale_pow}, fid={fid} \n')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")
flags.DEFINE_float("cfg", -1, 'cfg')
flags.DEFINE_float("cfg_scale_pow", -1, 'cfg decay')
flags.DEFINE_string("IMGsave_path", None, "The path to image log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.IMGsave_path = FLAGS.IMGsave_path
    config.cfg = FLAGS.cfg
    config.cfg_scale_pow = FLAGS.cfg_scale_pow
    
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
