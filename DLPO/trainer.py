from lightning_model import Wavegrad2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
from pytorch_lightning import Callback
import torch
from pytorch_lightning.utilities import rank_zero_only
from copy import deepcopy
from utils.tblogger import TensorBoardLoggerExpanded
from pathlib import Path
# from MOSNet.model import CNN_BLSTM
from NISQA.nisqa.NISQA_model import nisqaModel

from UTMOS.lightning_module import BaselineLightningModule
import logging
import time
from buffer_jc import RolloutBuffer
from datamodule import DataModule
import torchaudio

# torch.multiprocessing.set_start_method('spawn')


# Other DDPM/Score-based model applied EMA
# In our works, there are no significant difference
class EMACallback(Callback):

    def __init__(self, filepath, alpha=0.999, k=3):
        super().__init__()
        self.alpha = alpha
        self.filepath = filepath
        self.k = 3  #max_save
        self.queue = []
        self.last_parameters = None

    @rank_zero_only
    def _del_model(self, removek):
        if os.path.exists(self.filepath.format(epoch=removek)):
            os.remove(self.filepath.format(epoch=removek))

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx,
    ):
        if hasattr(self, 'current_parameters'):
            self.last_parameters = self.current_parameters
        else:
            self.last_parameters = deepcopy(pl_module.state_dict())

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        self.current_parameters = deepcopy(pl_module.state_dict())
        for k, v in self.current_parameters.items():
            self.current_parameters[k].copy_(self.alpha * v +
                                             (1. - self.alpha) *
                                             self.last_parameters[k])
        del self.last_parameters
        return

    @rank_zero_only
    def on_tarin_epoch_end(self, trainer, pl_module):
        if hasattr(self, 'current_parameters'):
            self.queue.append(trainer.current_epoch)
            torch.save(self.current_parameters,
                       self.filepath.format(epoch=trainer.current_epoch))
            pl_module.print(
                f"{self.filepath.format(epoch = trainer.current_epoch)} is saved"
            )

            while len(self.queue) > self.k:
                self._del_model(self.queue.pop(0))
        else:
            self.current_parameters = deepcopy(pl_module.state_dict())

        return


cwd = Path.cwd()


def load_mosnet(init_mostnet, model_dir: Path):
    if (model_dir.exists()):
        model_list = list(model_dir.iterdir())
    else:
        model_list = []

    if len(model_list) > 0:

        last_model = model_list[-1]

        last_epoch = int(last_model.stem)

        print(f"load model from epoch {last_epoch} from {last_model}")

        # TODO: load model parameters instead of model object
        init_mostnet.load_state_dict(torch.load(last_model))
        last_epoch = int(last_model.stem)
    else:
        last_epoch = 0
    return last_epoch


def train(args):
    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    datamodule = DataModule(cwd / "buffer1", hparams)
    datamodule.setup("")

    print(torch.cuda.is_available())
    tblogger = TensorBoardLoggerExpanded(hparams)
    ckpt_path = f'{hparams.log.name}_{now}_{args.loss_type}_{{epoch}}_{{loss}}test'
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.log.checkpoint_dir,
        filename=ckpt_path,
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='loss',
        mode='min',
        every_n_epochs=1,
    )
    trainer = Trainer(
        devices=hparams.train.num_workers,
        # devices=-1,
        # gpus=hparams.train.gpus,
        # accelerator="auto",
        # precision="16-mixed",
        # precision="bf16-mixed",
        precision="32",
        # accelerator='ddp' if hparams.train.gpus >= 1 else None,
        accelerator='gpu' if (hparams.train.num_workers >= 1
                              and torch.cuda.is_available()) else 'cpu',
        strategy='ddp_find_unused_parameters_true',
        # strategy=DDPStrategy(find_unused_parameters=True),
        #plugins='ddp_sharded',
        # amp_backend='apex',  #
        # amp_level='O2',  #
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        max_epochs=20000,
        logger=tblogger,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EMACallback(
                os.path.join(hparams.log.checkpoint_dir,
                             f'{hparams.name}_epoch={{epoch}}_EMA')),
            checkpoint_callback,
        ],
        num_nodes=hparams.train.nodes,
        val_check_interval=1.0,
        log_every_n_steps=1,
    )
    
    nisqa=nisqaModel(args)
    
    utmos=BaselineLightningModule.load_from_checkpoint(
            "/users/PAS2062/delijingyic/project/wavegrad2/UTMOS/epoch=3-step=7459.ckpt",map_location='cpu').eval()
    
    resampler = torchaudio.transforms.Resample(
            orig_freq=22050,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        )

    model_org = Wavegrad2(
        hparams,
        buffer=datamodule.buffer,
        train=args,
        get_ref_model=lambda : None,
        get_nisqa=lambda : None,
        get_resampler=lambda : None,
        get_utmos=lambda : None,
    )
    move_resampler = lambda device: resampler.to(device)
    call_resampler = lambda x: resampler.forward(x)

    move_modelorg = lambda device: model_org.to(device)
    call_modelorg = lambda text_org, frame, duration_target, speakers, input_lengths, output_lengths, noise_level, wind_index: model_org.forward(
        text_org,
        frame,
        duration_target,
        speakers,
        input_lengths,
        output_lengths,
        noise_level,
        wind_index=wind_index,
    )

    move_modelorginf = lambda device: model_org.to(device)
    call_modelorginf = lambda text_rl,spk_id,hidden_rep, wav,eps_list,model_mean,pace: model_org.inference_rl_org(
        text_rl,
        spk_id,
        hidden_rep,
        wav,
        eps_list,
        model_mean,
        pace,
    )

    model = Wavegrad2(
        hparams,
        buffer=datamodule.buffer,
        train=args,
        get_ref_model=lambda : model_org,
        get_nisqa=lambda:nisqa,
        
        get_resampler=lambda:resampler,
        get_utmos=lambda:utmos,
    )


    if (hparams.train.num_workers > 0 and trainer.global_rank == 0):
        model = model.cuda()
        model_org = model_org.cuda()
        utmos=utmos.cuda()
        resampler=resampler.cuda()

    OUTPUT_DIR = './output'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


    if args.ema:
        ckpt = torch.load(glob(
            os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}_EMA'))[-1],
                          map_location='cpu')

        sd = model.state_dict()

        for k, v in sd.items():
            if k in ckpt:
                if ckpt[k].shape == v.shape:
                    sd[k].copy_(ckpt[k])
        args.resume_from = None

    baseckpath = None if args.resume_from == None or args.restart else sorted(
        Path(hparams.log.checkpoint_dir).glob(
            f'*_epoch=1059_base.ckpt'))[-1]
    
    ckpath = None if args.resume_from == None or args.restart else sorted(
        Path(hparams.log.checkpoint_dir).glob(
            f'*_epoch={args.resume_from}_base.ckpt'))[-1]
    

    print(f"Loading Model {ckpath}")
    print(f"Trainer.global_rank {trainer.global_rank}")
    if (len(datamodule.buffer) <= 0 and trainer.global_rank == 0):
        # print('runif')

        model.load_state_dict(torch.load(ckpath)["state_dict"])
        model_org.load_state_dict(torch.load(baseckpath)["state_dict"])
        
        for name, params in model.named_parameters():
            assert not (torch.any(torch.isnan(params))
                        or torch.any(torch.isinf(params))), name
        for batch in datamodule.val_dataloader():
            model.validation_step(batch, 0)
            break
    st = torch.load(ckpath, map_location="cpu")["state_dict"]
    st_org = torch.load(baseckpath, map_location="cpu")["state_dict"]
    
    model.load_state_dict(st)
    model_org.load_state_dict(st_org)

    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,\
            required = False, help = "Resume Checkpoint epoch number")
    parser.add_argument('-s', '--restart', action = "store_true",\
            required = False, help = "Significant change occured, use this")
    parser.add_argument('-e', '--ema', action = "store_true",\
            required = False, help = "Start from ema checkpoint")
    parser.add_argument("--save_model_dir", type=str, default=None)

    parser.add_argument("--save_audio_dir", type=str, default=None)
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device, 'cuda' or 'cpu'")
    parser.add_argument("--loss_type", type=str, default=None)
    parser.add_argument('--mode', required=True, type=str, help='either predict_file, predict_dir, or predict_csv')
    parser.add_argument('--pretrained_model', required=True, type=str, help='file name of pretrained model (must be in current working folder)')
    parser.add_argument('--deg', type=str, help='path to speech file')
    parser.add_argument('--data_dir', type=str, help='folder with speech files')
    parser.add_argument('--output_dir', type=str, help='folder to ouput results.csv')
    parser.add_argument('--csv_file', type=str, help='file name of csv (must be in current working folder)')
    parser.add_argument('--csv_deg', type=str, help='column in csv with files name/path')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for pytorchs dataloader')
    parser.add_argument('--bs', type=int, default=1, help='batch size for predicting')
    parser.add_argument('--ms_channel', type=int, help='audio channel in case of stereo file')
    
    args = parser.parse_args()
    for x in ["save_audio_dir", "save_model_dir"]:
        if getattr(args, x) is not None:
            setattr(args, x, Path(getattr(args, x)))
    train(args)
