from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.io.wavfile import write as swrite
import numpy as np
import math
from omegaconf import OmegaConf as OC
from text import Language
from model.encoder import TextEncoder
from model.resampling import Resampling
from model.window import Window
from dataloader import TextAudioDataset
import re
from g2p_en import G2p
from tqdm import tqdm
from buffer_jc import RolloutBuffer
from pytorch_lightning.utilities import grad_norm
from pathlib import Path
import torch.nn.functional as F
from utils.utils import pad_list_dim2
from torch.distributions.normal import Normal
from NISQA.nisqa.NISQA_model import nisqaModel

from UTMOS.lightning_module import BaselineLightningModule
from torchaudio.transforms import Resample
import json



cwd = Path.cwd()
class Wavegrad2(pl.LightningModule):

    def __init__(
        self,
        hparams,
        get_ref_model:Callable[[],'Wavegrad2'],
        get_nisqa:Callable[[],nisqaModel],
        get_utmos:Callable[[],BaselineLightningModule],
        get_resampler:Callable[[],Resample],
        buffer: RolloutBuffer,
        train=True,
        
        
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.scale = hparams.window.scale
        self.symbols = Language(hparams.data.lang,
                                hparams.data.text_cleaners).get_symbols()
        self.symbols = ['"{}"'.format(symbol) for symbol in self.symbols]
        self.encoder = TextEncoder(hparams.encoder.channel,
                                   hparams.encoder.kernel,
                                   hparams.encoder.depth,
                                   hparams.encoder.dropout_rate,
                                   len(self.symbols))
        self.speaker_embedding = nn.Embedding(len(hparams.data.speakers),
                                              hparams.encoder.speaker_emb)
        self.resampling = Resampling(hparams)

        self.window = Window(hparams)
        if hparams.wavegrad.is_large:
            from model.nn_large import WaveGradNN
        else:
            from model.nn import WaveGradNN
        self.decoder = WaveGradNN(hparams)
        self.norm = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.set_noise_schedule(hparams, train)
        self.hparams_rl = OC.load('hparameter2.yaml')
        self.buffer = buffer
        self.traindata = TextAudioDataset(self.hparams_rl,
                                          self.hparams_rl.data.train_dir,
                                          self.hparams_rl.data.train_meta,
                                          train=False)

        self.get_nisqa=get_nisqa
        self.get_utmos=get_utmos
        self.get_resampler=get_resampler
        self.get_ref_model=get_ref_model
        # self.automatic_optimization = False
        # self.automatic_optimization=False
    # DDPM backbone is adopted form https://github.com/ivanvovk/WaveGrad

    @property
    def resampler(self):
        return self.get_resampler()
    @property
    def nisqa(self):
        return self.get_nisqa()
    @property
    def utmos(self):
        return self.get_utmos()
    @property
    def ref_model(self):
        return self.get_ref_model()
    
    def _get_variance_logprob(self, timestep, prev_timestep):
        # if timestep <= 1:
        #     return torch.tensor(1e-8, device=self.device)
        alpha_prod_t = self.alphas_cumprod[timestep].to(self.device)
        mask_a = int(prev_timestep >= 0)
        mask_a=torch.tensor(mask_a).to(self.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep].to(self.device) * mask_a
            + self.alphas_cumprod[0].to(self.device) * mask_b
        )
        beta_prod_t = torch.clamp(1 - alpha_prod_t, min=1e-6)
        beta_prod_t_prev = torch.clamp(1 - alpha_prod_t_prev, min=1e-6)
        variance = (torch.clamp(beta_prod_t_prev, min=1e-6) / torch.clamp(beta_prod_t, min=1e-6)) * (
    1 - alpha_prod_t / alpha_prod_t_prev
    
)       
        variance = torch.clamp(variance, min=1e-8) 
        

        return variance
    def step_logprob(
      self,
      model_output,
      timestep,
      sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic
        """Predict the sample at the previous timestep by reversing the SDE.

        Core function to propagate the diffusion process from the learned model
        outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion
            model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`): current instance of sample being created
            by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected"
            `model_output` from the clipped predicted original sample. Necessary
            because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened,
            "corrected" `model_output` would coincide with the one provided as
            input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for
            the variance using `generator`, we can directly provide the noise for
            the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than
            DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

            log_prob (`torch.FloatTensor`): log probability of the sample.
        """

        # pylint: disable=line-too-long
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        log_prob_list=[]
    
        prev_timestep = (
            timestep - 1000 // 1000
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].to(self.device)
        mask_a = int(prev_timestep >= 0)
        mask_a=torch.tensor(mask_a).to(self.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep].to(self.device) * mask_a
            + self.alphas_cumprod[0].to(self.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        # if self.config.clip_sample:
        #     pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance_logprob(timestep, prev_timestep).to(
            dtype=sample.dtype
        )
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)
        
        if use_clipped_model_output:
        # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output
        dim_0_size = pred_sample_direction.shape[0]
        std_dev_t = std_dev_t.expand(dim_0_size).unsqueeze(-1)
        variance = variance.expand(dim_0_size).unsqueeze(-1)
        # pylint: disable=line-too-long
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure"
                    " that either `generator` or `variance_noise` stays `None`."
                )

        if variance_noise is None:
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
        variance = std_dev_t * variance_noise
        dist = Normal(prev_sample, std_dev_t)
        prev_sample = prev_sample + variance
        log_prob = (
            dist.log_prob(prev_sample)
            .mean(dim=-1)

            
        )
        log_prob_list.append(log_prob)
        log_prob_list = torch.stack(log_prob_list)
        

        return log_prob_list
    
    def step_forward_logprob(
      self,
      model_output,
      timestep_t,
      sample,
      next_sample,
      eta = 1.0,
      use_clipped_model_output = False,
      generator=None,
      variance_noise = None,
      return_dict = True,
  ):  # pylint: disable=g-bare-generic

        """Predict the sample at the previous timestep by reversing the SDE.

        Core function to propagate the diffusion process from the learned model
        outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion
            model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`): current instance of sample (x_t) being
            created by diffusion process.
            next_sample (`torch.FloatTensor`): instance of next sample (x_t-1) being
            created by diffusion process. RL sampling is the backward process,
            therefore, x_t-1 is the "next" sample of x_t.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected"
            `model_output` from the clipped predicted original sample. Necessary
            because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened,
            "corrected" `model_output` would coincide with the one provided as
            input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for
            the variance using `generator`, we can directly provide the noise for
            the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than
            DDIMSchedulerOutput class

        Returns:
            log probability.
        """
        # if self.num_inference_steps is None:
        # raise ValueError(
        #     "Number of inference steps is 'None', you need to run 'set_timesteps'"
        #     " after creating the scheduler"
        # )

        # pylint: disable=line-too-long
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        log_prob_list=[]
        
        prev_timestep =(timestep_t - 1000 // 1000)
        print('prev_timestep',prev_timestep)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep_t].to(self.device)
        mask_a = int(prev_timestep >= 0)
        mask_a=torch.tensor(mask_a).to(self.device)
        mask_b = 1 - mask_a
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep].to(self.device) * mask_a
            + self.alphas_cumprod[0].to(self.device) * mask_b
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        

        # 4. Clip "predicted x_0"
        # if self.config.clip_sample:
        # pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)

        variance = self._get_variance_logprob(timestep_t, prev_timestep).to(
            dtype=sample.dtype
        )
          # Make sure variance has shape [dim_0_size]
        
        std_dev_t = (eta * variance ** (0.5)).to(dtype=sample.dtype)
        
        

        if use_clipped_model_output:
        # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output
        dim_0_size = pred_sample_direction.shape[0]
        std_dev_t = std_dev_t.expand(dim_0_size).unsqueeze(-1)
        variance = variance.expand(dim_0_size).unsqueeze(-1)
        
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction
        )
        

        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure"
                    " that either `generator` or `variance_noise` stays `None`."
                )

        if variance_noise is None:
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
        variance = std_dev_t * variance_noise
        dist = Normal(prev_sample, std_dev_t)
        
        log_prob = (
            dist.log_prob(next_sample)
            .mean(dim=-1)
        )
        
        log_prob_list.append(log_prob)
        log_prob_list = torch.stack(log_prob_list)
        
        return log_prob_list
    
    def set_noise_schedule(self, hparams, train=True):
        """
        Sets sampling noise schedule. Authors in the paper showed
        that WaveGrad supports variable noise schedules during inference.
        Thanks to the continuous noise level conditioning.
        :param init (callable function, optional): function which initializes betas
        :param init_kwargs (dict, optional): dict of arguments to be pushed to `init` function.
            Should always contain the key `steps` corresponding to the number of iterations to be done by the model.
            This is done so because `torch.linspace` has this argument named as `steps`.
        """
        self.max_step = hparams.ddpm.max_step if train \
            else hparams.ddpm.infer_step
        noise_schedule = eval(hparams.ddpm.noise_schedule) if train \
            else eval(hparams.ddpm.infer_schedule)

        # Calculations for posterior q(y_n|y_0)
        self.register_buffer('betas', noise_schedule, False)
        self.register_buffer('alphas', 1 - self.betas, False)
        self.register_buffer('alphas_cumprod', self.alphas.cumprod(dim=0),
                             False)
        self.register_buffer(
            'alphas_cumprod_prev',
            torch.cat([torch.FloatTensor([1.]), self.alphas_cumprod[:-1]]),
            False)
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.]), self.alphas_cumprod])
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             alphas_cumprod_prev_with_last.sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod', self.alphas_cumprod.sqrt(),
                             False)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             (1. / self.alphas_cumprod).sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod_m1',
                             (1. - self.alphas_cumprod).sqrt() *
                             self.sqrt_recip_alphas_cumprod, False)
        
        # Calculations for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) \
                             / (1 - self.alphas_cumprod)
        posterior_variance = torch.stack(
            [posterior_variance,
             torch.FloatTensor([1e-20] * self.max_step)])
        posterior_log_variance_clipped = posterior_variance.max(
            dim=0).values.log()
        posterior_mean_coef1 = self.betas * self.alphas_cumprod_prev.sqrt() / (
            1 - self.alphas_cumprod)
        posterior_mean_coef2 = (1 - self.alphas_cumprod_prev
                                ) * self.alphas.sqrt() / (1 -
                                                          self.alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_log_variance_clipped, False)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1,
                             False)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2,
                             False)

    def sample_continuous_noise_level(self, step):
        """
        Samples continuous noise level sqrt(alpha_cumprod).
        This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models.
        """
        step = torch.tensor(step, device=self.device)
        rand = torch.rand_like(step, dtype=torch.float, device=self.device)
        continuous_sqrt_alpha_cumprod = \
            self.sqrt_alphas_cumprod_prev[step - 1] * rand \
            + self.sqrt_alphas_cumprod_prev[step] * (1. - rand)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)

    def q_sample(self, y_0, step=None, noise_level=None, eps=None):
        """
        Efficiently computes diffusion version y_t from y_0 using a closed form expression:
            y_t = sqrt(alpha_cumprod)_t * y_0 + sqrt(1 - alpha_cumprod_t) * eps,
            where eps is sampled from a standard Gaussian.
        """
        batch_size = y_0.shape[0]
        if noise_level is not None:
            continuous_sqrt_alpha_cumprod = noise_level
        elif step is not None:
            continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[step]
        assert (step is not None or noise_level is not None)
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0, device=y_0.device, dtype=torch.float32)  
        
        y_0=y_0.unsqueeze(1)
        eps=eps.unsqueeze(1)
        
        continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.unsqueeze(-1)
        
        outputs = continuous_sqrt_alpha_cumprod * y_0 + (
            1. - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        return outputs
    ##no gradient
    # @torch.no_grad()
    def q_posterior(self, y_0, y, step):
        """
        Computes reverse (denoising) process posterior q(y_{t-1}|y_0, y_t, x)
        parameters: mean and variance.
        """

        pmean0 = torch.mul(y_0, self.posterior_mean_coef1[step].unsqueeze(-1))
        pmean1 = torch.mul(y, self.posterior_mean_coef2[step].unsqueeze(-1))
        posterior_mean = pmean0 + pmean1
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[
            step]
        return posterior_mean, posterior_log_variance_clipped

    def predict_start_from_noise_updateloss(self, y, t, eps):
        """
        Computes y_0 from given y_t and reconstructed noise.
        Is needed to reconstruct the reverse (denoising)
        process posterior q(y_{t-1}|y_0, y_t, x).
        """
        return self.sqrt_recip_alphas_cumprod[t].unsqueeze(
            -1) * y - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps

    @torch.no_grad()
    def predict_start_from_noise(self, y, t, eps):
        """
        Computes y_0 from given y_t and reconstructed noise.
        Is needed to reconstruct the reverse (denoising)
        process posterior q(y_{t-1}|y_0, y_t, x).
        """
        return self.sqrt_recip_alphas_cumprod[t].unsqueeze(
            -1) * y - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps

    ###no gradient
    # @torch.no_grad()
    def p_mean_variance_updateloss(self, y, hidden_rep, t, clip_denoised):
        """
        Computes Gaussian transitions of Markov chain at step t
        for further computation of y_{t-1} given current state y_t and features.
        """
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].unsqueeze(-1)
        eps_recon = self.decoder.forward(y, hidden_rep, noise_level)
        y_recon = self.predict_start_from_noise_updateloss(y, t, eps_recon)
        
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, y, hidden_rep, t, clip_denoised):
        """
        Computes Gaussian transitions of Markov chain at step t
        for further computation of y_{t-1} given current state y_t and features.
        """
        batch_size = y.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(
            batch_size, 1)
        eps_recon = self.decoder(y, hidden_rep, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def compute_inverse_dynamincs(self, y, hidden_rep, t, clip_denoised=False):
        """
        Computes reverse (denoising) process dynamics. Closely related to the idea of Langevin dynamics.
        :param y (torch.Tensor): previous state from dynamics trajectory
        :param clip_denoised (bool, optional): clip signal to [-1, 1]
        :return (torch.Tensor): next state
        """
        model_mean, model_log_variance = self.p_mean_variance(
            y, hidden_rep, t, clip_denoised)

        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    # @torch.no_grad()
    def compute_inverse_dynamincs_log_prob(
        self,
        y,
        hidden_rep,
        t,
        clip_denoised=False,
    ):
        """
        Computes reverse (denoising) process dynamics. Closely related to the idea of Langevin dynamics.
        :param y (torch.Tensor): previous state from dynamics trajectory
        :param clip_denoised (bool, optional): clip signal to [-1, 1]
        :return (torch.Tensor): next state
        """
        ###no gradient
        model_mean, model_log_variance = self.p_mean_variance(
            y, hidden_rep, t, clip_denoised)

        model_std = (0.5 * model_log_variance).exp().unsqueeze(-1)
        eps = torch.randn_like(y)
        eps[t <= 0] = 0
        model_sample = model_mean + eps * model_std

        log_prob = (-((model_sample.detach() - model_mean)**2) /
                    (2 * (model_std**2)) - torch.log(model_std) -
                    torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return model_sample, log_prob

    @torch.no_grad()
    def sample_rl(
        self,
        hidden_rep,
        start_step=None,
        store_intermediate_states=False,
    ):
        """
        Samples speech waveform via progressive denoising of white noise with guidance of mels-epctrogram.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional): whether to store dynamics trajectory or not
        :return ys (list of torch.Tensor) (if store_intermediate_states=True)
            or y_0 (torch.Tensor): predicted signals on every dynamics iteration of shape [B, T]
        """
        batch_size, T = hidden_rep.shape[0], hidden_rep.shape[-1]
        start_step = self.max_step if start_step is None \
            else min(start_step, self.max_step)
        y_t = torch.randn(batch_size, T * self.scale, device=self.device)
        ys = [y_t]
        ys_ref=[y_t]
        log_probs = []
        log_probs_ref = []

        model_means = []
        model_stds = []
        model_stds_ref = []

        eps_list=[]
        t = start_step - 1  #

        while t >= 0:

            noise_level = self.sqrt_alphas_cumprod_prev[t + 1].unsqueeze(-1)

            eps_recon = self.decoder.forward(y_t, hidden_rep, noise_level)
            eps_recon_ref = self.ref_model.decoder.forward(y_t, hidden_rep, noise_level)
            y_recon = self.predict_start_from_noise(y_t, t, eps_recon)
            y_recon_ref = self.predict_start_from_noise(y_t, t, eps_recon_ref)
            
        
            y_recon.clamp_(-1.0, 1.0)
            y_recon_ref.clamp_(-1.0, 1.0)
            
            model_mean, model_log_variance = self.q_posterior(
                y_recon, y_t, t)
            model_mean_ref, model_log_var_ref = self.ref_model.q_posterior(
                y_recon_ref, y_t, t)

            model_std = (0.5 * model_log_variance).exp().unsqueeze(-1)
            model_std_ref = (0.5 * model_log_var_ref).exp().unsqueeze(-1)

            
            eps = torch.randn_like(y_t)
            eps[t <= 0] = 0
            eps_list.append(eps)
            model_sample = model_mean + eps * model_std
            model_sample_ref = model_mean_ref + eps * model_std_ref
            log_prob_t=self.step_forward_logprob(eps_recon,t,y_recon,model_sample)
            log_prob_t=torch.tensor([log_prob_t[0].item()])
            log_prob_t_ref=self.step_forward_logprob(eps_recon_ref,t,y_recon_ref,model_sample_ref)
            log_prob_t_ref=torch.tensor([log_prob_t_ref[0].item()])
            y_t = model_sample
            y_t_ref=model_sample_ref

            ys.append(y_t)
            ys_ref.append(y_t_ref)
            log_probs.append(log_prob_t)
            log_probs_ref.append(log_prob_t_ref)

            model_means.append(model_mean)
            model_stds.append(model_std)
            model_stds_ref.append(model_std_ref)
            t -= 1

        if store_intermediate_states:
            wv: torch.Tensor = torch.stack(ys)
            wv_ref=torch.stack(ys_ref)
            log_probs = torch.stack(log_probs)
            log_probs_ref = torch.stack(log_probs_ref)


        else:
            wv = ys[-1]
            log_probs = log_probs[-1]
        
        return wv, log_prob_t, log_probs, model_means, model_stds,log_probs_ref,wv_ref,model_stds_ref
    
    def sample_rl_org(
        self,
        hidden_rep,
        wav,
        eps_list,
        model_mean_theta,
        start_step=None,
        store_intermediate_states=False,
    ):
        """
        Samples speech waveform via progressive denoising of white noise with guidance of mels-epctrogram.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional): whether to store dynamics trajectory or not
        :return ys (list of torch.Tensor) (if store_intermediate_states=True)
            or y_0 (torch.Tensor): predicted signals on every dynamics iteration of shape [B, T]
        """
        batch_size, T = hidden_rep.shape[0], hidden_rep.shape[-1]
        start_step = self.max_step if start_step is None \
            else min(start_step, self.max_step)
        y_t = torch.randn(batch_size, T * self.scale, device=self.device)
        ys = [y_t]
        log_probs = []
        model_means = []
        model_stds = []
        t = start_step - 1  #
        while t >= 0:
            model_mean, model_log_variance = self.p_mean_variance_updateloss(
                wav[start_step-1-t], hidden_rep, t, clip_denoised=True)

            model_std = (0.5 * model_log_variance).exp().unsqueeze(-1)
            eps=eps_list[start_step-1-t]
            model_sample = model_mean + eps * model_std

            log_prob_t = (-((wav[start_step-1-t+1].detach() - model_mean)**2) /
                          (2 * (model_std**2)) - torch.log(model_std) -
                          torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
            y_t = model_sample
            ys.append(y_t)
            log_probs.append(log_prob_t)
            model_means.append(model_mean)
            model_stds.append(model_std)
            t -= 1

        if store_intermediate_states:
            wv = torch.stack(ys)
            log_probs = torch.stack(log_probs)

        else:
            wv = ys[-1]
            log_probs = log_probs[-1]
        return wv, log_prob_t, log_probs, model_means, model_stds

    @torch.no_grad()
    def sample(self,
               hidden_rep,
               start_step=None,
               store_intermediate_states=False):
        """
        Samples speech waveform via progressive denoising of white noise with 
        :param store_intermediate_states (bool, optional): whether to store dynamics trajectory or not
        :return ys (list of torch.Tensor) (if store_intermediate_states=True)
            or y_0 (torch.Tensor): predicted signals on every dynamics iteration of shape [B, T]
        """
        batch_size, T = hidden_rep.shape[0], hidden_rep.shape[-1]
        start_step = self.max_step if start_step is None \
            else min(start_step, self.max_step)
        y_t = torch.randn(batch_size, T * self.scale, device=self.device)
        ys = [y_t]
        t = start_step - 1  #
        pbar = tqdm(total=t)
        while t >= 0:
            y_t = self.compute_inverse_dynamincs(
                y_t,
                hidden_rep,
                t,
                clip_denoised=True,
            )
            ys.append(y_t)

            t -= 1
            pbar.update()

        return ys if store_intermediate_states else ys[-1]

    def forward(
        self,
        wav_noisy_sliced,
        hidden_rep_sliced,
        noise_level,
        t,
        wind_index,
        no_mask=False,
    ):
        # [B, N, chn.encoder]
        hidden_rep_sliced = hidden_rep_sliced.to(self.device)
        wav_noisy_sliced = wav_noisy_sliced.to(self.device)
        noise_level = noise_level.to(self.device)
        wind_index = wind_index.to(self.device)
        
        # log_prob_list=[]
        # log_prob_list_cur=[]
        
        
        eps_recon = self.decoder.forward(
                wav_noisy_sliced,
                hidden_rep_sliced,
                noise_level,
            )
            
        y_recon = self.predict_start_from_noise(wav_noisy_sliced, t, eps_recon)
        y_recon.clamp_(-1.0, 1.0)
        model_mean, model_log_variance = self.q_posterior(
            y_recon, wav_noisy_sliced, t)
        model_std = (0.5 * model_log_variance).exp().unsqueeze(-1)
        eps1 = torch.randn_like(wav_noisy_sliced)
        eps1[t <= 0] = 0
        model_sample = model_mean + eps1 * model_std
        log_prob= self.step_forward_logprob(eps_recon,t,wav_noisy_sliced,model_sample)
        log_prob_cur= self.step_logprob(eps_recon,t,wav_noisy_sliced)
        
        return eps_recon, log_prob,log_prob_cur 
        
    def inference(self, text, speakers, pace=1.0):
        text_encoding = self.encoder.inference(text)

        speaker_emb = self.speaker_embedding(speakers)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1,
                                                      text_encoding.size(1),
                                                      -1)

        decoder_input = torch.cat((text_encoding, speaker_emb), dim=2)
        hidden_rep, alignment, duration, sigma = self.resampling.inference(
            decoder_input, pace=pace)

        wav_recon = self.sample(hidden_rep, store_intermediate_states=False)
        wav_recon = torch.clamp(wav_recon,
                                min=-1,
                                max=1 - torch.finfo(torch.float16).eps)
        return wav_recon, alignment, duration, sigma

        

    def inference_rl(self, text, speakers, hidden_rep, pace=1.0):
        
        wav_recon, log_prob_t, log_probs, model_mean, model_std,log_probs_ref,wv_ref,model_stds_ref = self.sample_rl(
            hidden_rep,
            store_intermediate_states=True,
        )

        wav_recon = torch.clamp(
            wav_recon,
            min=-1,
            max=1 - torch.finfo(torch.float16).eps,
        )
        return wav_recon, "", "", "", log_prob_t, log_probs, hidden_rep, model_mean, model_std,log_probs_ref,wv_ref,model_stds_ref

    def inference_rl_org(self, text, speakers, hidden_rep, wav,eps_list,model_means, pace=1.0):

        wav_recon, log_prob_t, log_probs, model_mean, model_std = self.sample_rl_org(
            hidden_rep,
            wav,
            eps_list,
            model_means,
            store_intermediate_states=True,
        )

        wav_recon = torch.clamp(
            wav_recon,
            min=-1,
            max=1 - torch.finfo(torch.float16).eps,
        )

        return wav_recon, "", "", "", log_prob_t, log_probs, hidden_rep, model_mean, model_std

    def read_lexicon(self, lex_path):
        lexicon = {}
        with open(lex_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def preprocess_eng(self, hparams, text):

        lexicon = self.read_lexicon(hparams.data.lexicon_path)
        g2p = G2p()
        phones_list = []
        words = re.split(r"([,;.\-\?\!\s+])", str(text))
        for w in words:
            if w.lower() in lexicon:
                if type(phones_list) == type("str"):
                    print("string phones", phones_list)
                for l in lexicon[w.lower()]:
                    phones_list.append(l)


            else:
                for l in list(filter(lambda p: p != " ", g2p(w))):
                    phones_list.append(l)
        phones_list = "{" + "}{".join(phones_list) + "}"
        phones_list = re.sub(r"\{[^\w\s]?\}", "{sp}", phones_list)
        text = self.traindata.get_text(phones_list)
        text = text.unsqueeze(0)
        return text

    def compose_wav(self, wav, step, eps):
        wav = wav.to(self.device)
        step = step.to(self.device)
        eps = eps.to(self.device)

        # #need
        y_recon = self.predict_start_from_noise_updateloss(
            wav,
            step,
            eps,
        )
        y_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon,
            wav,
            step,
        )
        return model_mean, posterior_log_variance_clipped
    def log_sum_exp(log_probs,dim=-1):
        max_log_prob = torch.max(log_probs, dim=dim, keepdim=True)
        return max_log_prob + torch.log(torch.sum(torch.exp(log_probs - max_log_prob), dim=dim))
    def cal_logp_entireaudio(self,wav_noisy_sliced,pwav_noisy_sliced,t,eps_recon,peps_recon):
        model_mean, posterior_log_variance_clipped = self.compose_wav(
                    wav=wav_noisy_sliced,
                    step=t,
                    eps=eps_recon,
                )
        pmean, pplvc = self.compose_wav(
            wav=pwav_noisy_sliced,
            step=t,
            eps=peps_recon,
        )
        
        model_std = (0.5 * posterior_log_variance_clipped).exp().unsqueeze(-1)
        pstd = (0.5 * pplvc).exp().unsqueeze(-1)
        
        wav_dist = Normal(
            model_mean,
            model_std,
        )
        p_dist = Normal(
            pmean,
            pstd,
        )

        model_sample = wav_dist.sample().clip(-1, 1)
        log_prob_t = wav_dist.log_prob(model_sample)
        

        log_prob_t_ddpo = (-((model_sample.detach() - model_mean)**2) /
                        (2 * (model_std**2)) - torch.log(model_std) -
                        torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
        plog_prob_t = p_dist.log_prob(model_sample)
        plog_prob_t_ddpo = (
            -((model_sample.detach() - pmean)**2) / (2 * (pstd**2)) -
            torch.log(pstd) -
            torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
        log_prob_t_ddpo_dim=torch.sum(log_prob_t_ddpo,dim=1)
        log_prob_t_ddpo = log_prob_t_ddpo.mean(
            dim=tuple(range(1, log_prob_t_ddpo.ndim)))

        plog_prob_t_ddpo_dim=torch.mean(plog_prob_t_ddpo,dim=1)
        plog_prob_t_ddpo = plog_prob_t_ddpo.sum(
            dim=tuple(range(1, plog_prob_t_ddpo.ndim)))
        log_prob_t = log_prob_t.mean(dim=tuple(range(1, log_prob_t.ndim)))
        plog_prob_t = plog_prob_t.mean(dim=tuple(range(1, plog_prob_t.ndim)))
        return log_prob_t,plog_prob_t,log_prob_t_ddpo,plog_prob_t_ddpo
    
    
    def calculate_noisy_audio(self,text_org,input_lengths,speakers,duration_target,output_lengths,goal_audio,noise_level):
        text_encoding = self.encoder.forward(
        text_org,
        input_lengths,
        )
        speaker_emb = self.speaker_embedding.forward(
            speakers)  # [B, chn.speaker]
        speaker_emb = speaker_emb.unsqueeze(1).expand(
            -1, text_encoding.size(1), -1)  # [B, N, chn.speaker]

        decoder_input = torch.cat((text_encoding, speaker_emb), dim=2)
        hidden_rep, alignment, duration, mask = self.resampling.forward(
            decoder_input,
            duration_target,
            input_lengths,
            output_lengths,
            no_mask=False,
        )

        wav_sliced, hidden_rep_sliced = self.window(goal_audio, hidden_rep, output_lengths)
        
        eps = torch.randn_like(
            wav_sliced,
            device=self.device,
            dtype=torch.float32,
        )

        # eps=eps.unsqueeze(1)
        wav_noisy_sliced = self.q_sample(
            wav_sliced,
            noise_level=noise_level,
            eps=eps,
        )
        return wav_noisy_sliced, hidden_rep_sliced,eps,duration,mask
    
    def training_step(self, batch, batch_idx):
        frame, logprob, t, hidrep, mos, texts, mosscore_update, duration_target, speakers, input_lengths, output_lengths, text_org, model_mean_t, model_std_t, goal_audio, step_list, logprob_inf, nisqascore_update, rawtext = batch

        batch_size = goal_audio.shape[0]
        device = self.device
        max_step = 999
        sample_steps = 10
        step_decrement = max_step // sample_steps
        num_steps = sample_steps + 1  # Including t=0

        # Preallocate tensors for losses and log probabilities
        log_prob_list_allsteps = torch.empty((num_steps, batch_size), device=device)
        plog_prob_list_allsteps = torch.empty((num_steps, batch_size), device=device)
        l1_loss_list = torch.empty((num_steps, batch_size), device=device)
        duration_loss_list = torch.empty((num_steps, batch_size), device=device)
        kl_normalize_list = torch.empty((num_steps, batch_size), device=device)

        # Initialize wind_index tensor
        wind_index = torch.randint(0, max(output_lengths.max() - self.window.length, 1), (batch_size,), device=device)

        t1 = max_step
        step_idx = 0  # Keep track of steps

        while t1 >= 0:
            noise_level = self.sqrt_alphas_cumprod_prev[t1].unsqueeze(-1)
            wav_noisy_sliced, hidden_rep_sliced, eps, duration, mask = self.calculate_noisy_audio(
                text_org, input_lengths, speakers, duration_target, output_lengths, goal_audio, noise_level
            )

            eps_recon, log_prob_google, _ = self.forward(wav_noisy_sliced, hidden_rep_sliced, noise_level, t1, wind_index=wind_index)
            peps_recon, plog_prob_google, _ = self.ref_model.forward(wav_noisy_sliced, hidden_rep_sliced, noise_level, t1, wind_index=wind_index)

            # Compute losses
            l1_loss = self.norm(eps_recon, eps)
            l1_loss_list[step_idx] = l1_loss

            mask = ~mask
            duration_ = duration.masked_select(mask)
            duration_target_ = duration_target.masked_select(mask)
            duration_loss = self.mse_loss(duration_, duration_target_ / (self.hparams.audio.sampling_rate / self.hparams.window.scale))
            duration_loss_list[step_idx] = duration_loss

            kl_normalize_list[step_idx] = self.norm(eps_recon, peps_recon)

            # Store log probabilities
            log_prob_list_allsteps[step_idx] = log_prob_google
            plog_prob_list_allsteps[step_idx] = plog_prob_google

            t1 -= step_decrement
            step_idx += 1

        # Compute KL divergence and ratio
        kl1_google = log_prob_list_allsteps - plog_prob_list_allsteps
        kl1_google_all = logprob - logprob_inf
        ratio = torch.clamp(kl1_google, 1.0 - 1e-4, 1.0 + 1e-4).squeeze()
        diffusion_loss = l1_loss_list + duration_loss_list

        mosscore_update = mosscore_update.unsqueeze(0).unsqueeze(2).expand(num_steps, 2, 2)
        kl_normalize_list = kl_normalize_list.unsqueeze(1).unsqueeze(2).expand(num_steps, 2, 2)
        diffusion_loss = diffusion_loss.unsqueeze(1).unsqueeze(2).expand(num_steps, 2, 2)

        # ====== Loss functions (Uncomment the one you want to use) ======

        ## DPOK (Direct Preference Optimization with KL penalty)
        # loss = torch.mean(-ratio * mosscore_update + kl_normalize_list)

        ## DDPO (Diffusion-based Direct Preference Optimization)
        # loss = torch.mean(-ratio * mosscore_update)

        ## RWR (Reward-weighted regression)
        # loss = torch.mean(-log_prob_list_allsteps * mosscore_update)

        ## KLinR (KL-regularized reward optimization)
        # loss = torch.mean(-ratio * (mosscore_update - kl_normalize_list))

        ## Default: DLPO (Diffusion Loss Preference Optimization)
        loss = torch.mean(-ratio * (5 * mosscore_update - diffusion_loss))

        KL_diff_entire = torch.mean(logprob - logprob_inf)

        # Log losses
        loss_info = {
            "l1loss": torch.mean(l1_loss_list).cpu().detach().item(),
            "loss": loss.cpu().detach().item(),
            "mosscore": torch.mean(mosscore_update).cpu().detach().item(),
            "KL": torch.mean(kl_normalize_list).cpu().detach().item(),
            "KL_entire": KL_diff_entire.cpu().detach().item(),
            'nisqascore': torch.mean(nisqascore_update).cpu().detach().item(),
        }
        self.log_dict(loss_info)
        
        return loss





    def validation_step(self, batch, batch_idx):
        if self.global_rank != 0:
            return
        print("#" * 60)
        text_1, wav, duration_target, speakers, input_lengths, output_lengths, max_input_len, sorted_basename, mos_reward, filter_rawtext = batch

        
        with torch.no_grad():
            for i in range(len(filter_rawtext)):

                text_rl = self.preprocess_eng(
                    self.hparams_rl,
                    filter_rawtext[i],
                )
                text_rl = text_rl.to(self.device)

                speaker_dict = {
                    spk: idx
                    for idx, spk in enumerate(self.hparams_rl.data.speakers)
                }
                spk_id = [speaker_dict['LJSpeech']]
                spk_id = torch.LongTensor(spk_id)
                spk_id = spk_id.to(self.device)

                text_encoding = self.encoder.inference(text_rl)

                speaker_emb = self.speaker_embedding(spk_id)

                speaker_emb = speaker_emb.unsqueeze(1).expand(
                    -1, text_encoding.size(1), -1)

                decoder_input = torch.cat((text_encoding, speaker_emb), dim=2)
                with torch.no_grad():
                    hidden_rep, alignment, duration, sigma = self.resampling.inference(
                        decoder_input, pace=1.0)
                    wav_recon, alignment, duration, sigma, log_prob_1000, log_probs_sample, hdrep, model_mean, model_std,log_probs_ref,wv_ref,model_stds_ref  = self.inference_rl(
                        text_rl,
                        spk_id,
                        hidden_rep,
                        pace=1.0,
                    )

                audio = wav_recon[-1][0].float()
                audio_inf=wv_ref[-1][0].float()


                assert not torch.any(torch.isnan(audio)), f"audio has nan"

                
                ##UTMOS
                with torch.no_grad():
                    if len(audio.shape) == 1:
                        out_wavs = audio.unsqueeze(0).unsqueeze(0)
                    elif len(audio.shape) == 2:
                        out_wavs = audio.unsqueeze(0)
                    elif len(audio.shape) == 3:
                        out_wavs = audio
                    else:
                        raise ValueError('Dimension of input tensor needs to be <= 3.')
                    out_wavs = self.resampler.forward(out_wavs)
                    bs = out_wavs.shape[0]
                    batch = {
                        'wav': out_wavs,
                        'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
                        'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
                    }
                    bs = out_wavs.shape[0]
                    batch = {
                    'wav': out_wavs,
                    'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
                    'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
                    }
                    utmosscore = self.utmos.forward(batch)
                    utmosscore=utmosscore.mean(dim=1).squeeze(1).cpu().detach().numpy()*2 + 3
   
                audio_path = cwd / "mos_results" / "dlpo_999" / f"{self.global_step}_{filter_rawtext[:5]}_{utmosscore.item()}_test.wav"

                swrite(
                    audio_path,
                    22050,
                    audio_inf.detach().cpu().numpy(),
                )
                print('audio_path',audio_path)
                
                nisqascore=self.nisqa.predict_wg2(audio_path)    
                log_probs_ref=torch.mean(log_probs_ref)
                log_probs_sample=torch.mean(log_probs_sample)
                
                
                with open(
                    cwd/ "dpok11.cvs",
                    "ab") as f:
                    np.savetxt(f, np.array([log_probs_sample.item()]), delimiter=',')
                with open(
                    cwd/ "dpok11ref.cvs",
                    "ab") as f2:
                    np.savetxt(f2, np.array([log_probs_ref.item()]), delimiter=',')


                self.buffer.save(
                    hidrep=hidden_rep.to(torch.float).cpu().numpy(),
                    mosscore=mos_reward[i],
                    frame=wav_recon[1:].cpu().numpy(),
                    log_prob=log_probs_sample.cpu().numpy(),
                    text=text_rl.cpu().numpy(),
                    duration_target=duration_target.cpu().numpy(),
                    speakers=speakers.cpu().numpy(),
                    input_lengths=input_lengths.cpu().numpy(),
                    output_lengths=output_lengths.cpu().numpy(),
                    text_org=text_1[i].cpu().numpy(),
                    model_mean=torch.vstack(model_mean).cpu().numpy(),
                    model_std=torch.vstack(model_std).cpu().numpy(),
                    mosscore_update=utmosscore,
                    rawtext=filter_rawtext[i],
                    goal_audio=wav.cpu().numpy(),
                    log_probinf=log_probs_ref.cpu().numpy(),
                    model_stdinf=torch.vstack(model_stds_ref).cpu().numpy(),
                    nisqascore=nisqascore,


                 
                )


    def configure_optimizers(self):
        learnable_params = self.parameters()
        # assert False
        self.opt = torch.optim.Adam(
            learnable_params,
            lr=self.hparams.train.adam.lr,
            weight_decay=self.hparams.train.adam.weight_decay,
        )
        return self.opt

    def lr_lambda(self, step):
        progress = (step - self.hparams.train.decay.start) / (
            self.hparams.train.decay.end - self.hparams.train.decay.start)
        return self.hparams.train.decay.rate**np.clip(progress, 0.0, 1.0)

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_closure):
        lr_scale = self.lr_lambda(self.global_step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.hparams.train.adam.lr
        optimizer.step(closure=optimizer_closure)
        
        optimizer.zero_grad()

        self.trainer.logger.log_learning_rate(
            lr_scale * self.hparams.train.adam.lr, self.global_step)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = {
            key: val
            for key, val in grad_norm(self.encoder, norm_type=2).items()
        }

        norms.update({
            key: val
            for key, val in grad_norm(self.decoder, norm_type=2).items()
        })

        for key, val in norms.items():
            assert not (torch.isnan(val) or torch.isinf(val)), {
                key: val.item()
                for key, val in norms.items()
                if torch.isnan(val) or torch.isinf(val)
            }

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                continue


    def cuda(self, device: torch.device = None):
        mmm = super(__class__, self).cuda(device)
        for x in [self.utmos,self.resampler]:

            
            if(x is None):
                continue
            x.to(device)
        return mmm

    def to(self, device: torch.device = None):
        mmm = super(__class__, self).to(device)
        for x in [self.utmos,self.resampler]:
        
            if(x is None):
                continue
            x.to(device)

        return mmm


