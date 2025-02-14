from lightning_model import Wavegrad2
from omegaconf import OmegaConf as OC
import argparse
import datetime
import torch
from scipy.io.wavfile import write as swrite
from g2p_en import G2p
from pypinyin import pinyin, Style
import re
from dataloader import TextAudioDataset
from UTMOS import lightning_module
import torchaudio
from pathlib import Path
import json
from tqdm import tqdm
from glob import glob

import pandas as pd

cwd = Path.cwd()


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_eng(hparams, text):
    lexicon = read_lexicon(hparams.data.lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    print('g2p: ', phones)

    trainset = TextAudioDataset(hparams,
                                hparams.data.train_dir,
                                hparams.data.train_meta,
                                train=False)

    text = trainset.get_text(phones)
    text = text.unsqueeze(0)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Checkpoint path")
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only")
    parser.add_argument('--speaker',
                        type=str,
                        default='LJSpeech',
                        help="speaker name")
    parser.add_argument('--pace',
                        type=int,
                        default=1.0,
                        help="control the pace of the whole utterance")
    parser.add_argument('--steps',
                        type=int,
                        required=False,
                        help="Steps for sampling")
    parser.add_argument("--save_audio_dir", type=str, default=None)

    parser.add_argument('-r', '--resume_from', type =int,\
            required = False, help = "Resume Checkpoint epoch number")
    parser.add_argument('-f', '--filename', type=str,
                        default='filename', help = "Resume Checkpoint epoch number")
    args = parser.parse_args()

    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"

    ###load model
    checkpoint = args.checkpoint
    model = Wavegrad2(hparams, args).cuda()
    ckpt = torch.load(args.checkpoint)

    model.load_state_dict(ckpt['state_dict'] if not (
        'EMA' in args.checkpoint) else ckpt)
    model = model.cuda()

    sample_dir = Path(
        "/users/PAS2062/delijingyic/project/wavegrad2/dataset/LJSpeech")
    duration_dir = sample_dir / "preprocessed" / "duration"
    textgrid_dir = sample_dir / "preprocessed" / "TextGrid"

    utmos=lightning_module.BaselineLightningModule.load_from_checkpoint(
            "/users/PAS2062/delijingyic/project/wavegrad2/UTMOS/epoch=3-step=7459.ckpt",map_location='cpu').eval()
    
    resampler = torchaudio.transforms.Resample(
            orig_freq=22050,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        )
    utmos=utmos.cuda()
    resampler=resampler.cuda()
    textfile_path = Path(
        "/users/PAS2062/delijingyic/project/wavegrad2/dataset/LJSpeech/LJSpeech-1.1/inference_text2.csv"
    )
    data = pd.read_csv(textfile_path,
                       dtype=str,
                       sep='\t',
                       names=['filename', 'text'])
    mosscore = 0
    epoch_wg2 = checkpoint.split(".")[0]
    epoch_wg2 = epoch_wg2.split("=")[-1]
    file_name = f"MOS120_WG2{epoch_wg2}"
    try:
        with open(cwd / "mos_results" / f"{args.filename}_{file_name}.json",
                  "r") as load_f:
            mos_dic = json.loads(load_f.read())
            mosscore = mos_dic['mean_mos'] * len(mos_dic)
    except Exception:
        mos_dic = {}

    #get speaker id
    speaker_dict = {spk: idx for idx, spk in enumerate(hparams.data.speakers)}
    spk_id = torch.LongTensor([speaker_dict[args.speaker]]).cuda()

    #get path for output file
    result_dir = cwd / "mos_results" / f"{args.filename}_{file_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    mos_score_total = 0  # Track MOS score sum
    mos_count = 0  # Track valid samples

    for i, row in tqdm(data['text'].items()):
        filename = data.loc[i]['filename']
        
        # Skip if already processed
        if filename in mos_dic:
            continue

        # Process text only if needed
        text = preprocess_eng(hparams, row) if hparams.data.lang == 'eng' else row
        text = text.cuda()

        # Run model inference
        wav_recon, align, *_, timestep = model.inference(text, spk_id, pace=args.pace)

        # Save output audio
        swrite(result_dir / f"{filename}.wav", hparams.audio.sampling_rate, wav_recon[0].cpu().numpy())

        # Prepare waveform for UT-MOS scoring
        out_wavs = wav_recon.unsqueeze(0) if wav_recon.dim() < 3 else wav_recon
        out_wavs = resampler(out_wavs)

        # Prepare batch for UT-MOS
        batch = {
            'wav': out_wavs,
            'domains': torch.zeros(out_wavs.shape[0], dtype=torch.int, device=wav_recon.device),
            'judge_id': torch.full((out_wavs.shape[0],), 288, dtype=torch.int, device=wav_recon.device)
        }

        # Compute UT-MOS score
        utmosscore = utmos(batch).mean(dim=1).squeeze(1).cpu().numpy() * 2 + 3

        # Store MOS score
        mos_dic[filename] = utmosscore.tolist()
        mos_score_total += utmosscore[0]
        mos_count += 1

        # Update average MOS score
        mos_dic['mean_mos'] = mos_score_total / mos_count

        # Save MOS scores periodically to reduce I/O operations
        if i % 10 == 0 or i == len(data['text']) - 1:
            with open(result_dir / f"{args.filename}_{file_name}.json", "w") as f:
                json.dump(mos_dic, f)
