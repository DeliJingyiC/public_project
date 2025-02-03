import os
import numpy as np
import torch
from typing import Any, Dict, Tuple, Generator, List, Optional, Union
from pathlib import Path
from torch.utils.data import Dataset
from MOSNet.model import CNN_BLSTM
from torch.utils.data import DataLoader
import scipy
import random
import datetime
from torch.distributions.normal import Normal


def pad_lastdimension(x: List[np.ndarray], y: torch.Tensor):
    x = [k.squeeze() for k in x]
    assert np.all([k.shape[:-1] == x[0].shape[:-1]
                   for k in x]), f'{[k.shape for k in x]}'
    pad = np.zeros((
        len(x),
        *x[0].shape[:-1],
        max([k.shape[-1] for k in x]),
    ))
    for i, l in enumerate(x):
        pad[i, ..., :l.shape[-1]] = l
        pad[i, ..., l.shape[-1]:] = y[i]
    return pad


class RolloutBuffer(Dataset):

    def __init__(
        self,
        path: Path,
        buffer_size: int = 10,
    ):
        self.loaded = []
        self.buffer = []

        self.folder = path
        self.buffer_size = buffer_size

    def __len__(self):
        length = sum([len(i["frame"]) for i in self.buffer])
        return length

    def __getitem__(self, idex: int):
        ###test
        for i in range(len(self.buffer)):
            if idex < len(self.buffer[i]['frame']):
                if idex == len(self.buffer[i]['frame']) - 1:
                    idex = random.randint(len(self.buffer[i]['frame']) - 12, len(self.buffer[i]['frame']) - 2)
                    
                item_f = self.buffer[i]['frame'][idex]
                item_model_mean = self.buffer[i]['model_mean'][idex]
                item_model_std = self.buffer[i]['model_std'][idex]
                item_model_stdinf = self.buffer[i]['model_stdinf'][idex]
                step_list=[x+1 for x in range(len(self.buffer[i]['frame']))][-5:]
                
                return [
                    item_f,
                    self.buffer[i]['log_prob'],
                    idex + 1,
                    self.buffer[i]['hidrep'],
                    self.buffer[i]['mosscore'],
                    self.buffer[i]["text"],
                    self.buffer[i]['mosscore_update'],
                    self.buffer[i]['duration_target'],
                    self.buffer[i]['speakers'],
                    self.buffer[i]['input_lengths'],
                    self.buffer[i]['output_lengths'],
                    self.buffer[i]['text_org'],
                    item_model_mean,
                    item_model_std,
                    self.buffer[i]['rawtext'],
                    self.buffer[i]['goal_audio'],
                    step_list,
                    self.buffer[i]['log_probinf'],
                    item_model_stdinf,
                    self.buffer[i]['nisqascore'],
                    
                ]

            else:
                idex = idex - len(self.buffer[i]['frame'])

    def fetch_name(self, **data):
        eps_idx = len(list(self.folder.glob('*.dt')))
        random_idx = np.random.randint(0, 1000)

        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        text = data['text']
        eps_fn = f"{ts}_{eps_idx:09d}_{random_idx:04d}"
        full_path = self.folder / eps_fn
        return full_path


    def save(self, **data):
        file_path = self.fetch_name(**data)
        np.savez(
            file_path,
            **data,
        )
        file_path.with_suffix(".npz").rename(file_path.with_suffix(".dt"))
        print(f"Saved {file_path.with_suffix('.dt')}")
        return file_path



    def load_data(self):
        for x in sorted(self.folder.glob('*.dt')):
            if x in self.loaded:
                continue
            kk = np.load(x)

            self.buffer.append({**kk})
            self.loaded.append(x)

            print(f"Loaded {x} self.buffer {len(self.buffer)}")



        while (len(self.buffer) > self.buffer_size):
            print('before', [x['mosscore_update'] for x in self.buffer])
            self.buffer.pop(0)
            print('after', [x['mosscore_update'] for x in self.buffer])

    def add(
        self,
        hidrep,
        mosscore,
        frame,
        log_prob,
        text,
        duration_target,
        speakers,
        input_lengths,
        output_lengths,
        text_org,
        model_mean,
        model_std,
        mosscore_update,
        rawtext,
        goal_audio,
        step_list,
        log_probinf,
        item_model_stdinf,
        nisqascore,

    ):
        assert len(frame) == len(
            log_prob), f'frame {len(frame)}, log_prob {len(log_prob)}'
        print('mosscore_update buffer', mosscore_update)
        print(
            f"Adding {mosscore_update} to {[x['mosscore_update'] for x in self.buffer]}"
        )
        self.buffer.append({
            'hidrep': hidrep,
            'mosscore': mosscore,
            'frame': frame,
            'log_prob': log_prob,
            'text': text,
            "duration_target": duration_target,
            "speakers": speakers,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "text_org": text_org,
            "model_mean": model_mean,
            "model_std": model_std,
            'mosscore_update': mosscore_update,
            'rawtext': rawtext,
            'goal_audio': goal_audio,
            'step_list':step_list,
            'log_probinf': log_probinf,
            'item_model_stdinf':item_model_stdinf,
            'nisqascore':nisqascore,


        })

    def create_dataloader(self, batch_size, num_worker):

        self.load_data()

        def collate_fn(batch: List[Tuple[np.ndarray, float, int, np.ndarray,
                                         float]]):
            
            rawtext = [x[14] for x in batch]
            model_std = torch.tensor(np.array([x[13] for x in batch]))
            model_std_inf = torch.tensor(np.array([x[18] for x in batch]))
            head = batch[0]
            logprob = torch.tensor(np.array([x[1] for x in batch]))
            logprob_inf = torch.tensor(np.array([x[17] for x in batch]))
            tidx = torch.tensor([x[2] for x in batch])
            step_list = torch.tensor(np.array([x[16] for x in batch]))
            model_mean_lis = [x[12] for x in batch]
            model_mean = torch.tensor(
                pad_lastdimension(
                    model_mean_lis,
                    y=torch.zeros((len(model_mean_lis), )),
                ),
                dtype=torch.float32,
            )
            goal_audio_list = [x[15] for x in batch]
            goal_audio = pad_lastdimension(
                goal_audio_list,
                y=torch.zeros((len(goal_audio_list), )),
            )
            goal_audio = torch.tensor(goal_audio, dtype=torch.float32)

            ##pad frame
            frame_list = [x[0] for x in batch]
            paded_frame = pad_lastdimension(
                frame_list,
                y=torch.zeros((len(frame_list), )),
            )
            frame = torch.tensor(paded_frame, dtype=torch.float32)
            
            text_list = [x[5] for x in batch]
            paded_text = pad_lastdimension(
                text_list,
                y=torch.zeros((len(text_list), )),
            )
            texts = torch.tensor(paded_text, dtype=torch.long)
            
            hp_list = [x[3] for x in batch]
            paded_hp = pad_lastdimension(
                hp_list,
                y=torch.zeros((len(hp_list), )),
            )
            hidrep = torch.tensor(paded_hp)

            ##mosscore
            mos = torch.tensor(np.array([x[4] for x in batch]))
            mosscore_update = torch.tensor(np.array([x[6] for x in batch
                                                     ])).squeeze()
            
            duration_target = torch.tensor(
                pad_lastdimension(
                    [x[7] for x in batch],
                    y=torch.zeros((len(batch), )),
                ), ).float()
            speakers = torch.tensor(np.array([x[8] for x in batch])).squeeze()
            input_lengths = torch.tensor(np.array([x[9] for x in batch]),
                                         dtype=torch.int).squeeze()
            output_lengths = torch.tensor(np.array([x[10] for x in batch
                                                    ])).squeeze()
            text_org = torch.tensor(
                pad_lastdimension(
                    [x[11] for x in batch],
                    y=torch.zeros((len(batch), )),
                ),
                dtype=torch.long,
            )
            nisqascore_update = torch.tensor(np.array([x[19] for x in batch
                                                     ])).squeeze()
            kk = [
                frame,
                logprob,
                tidx,
                hidrep,
                mos,
                texts,
                mosscore_update,
                duration_target,
                speakers,
                input_lengths,
                output_lengths,
                text_org,
                model_mean,
                model_std,
                goal_audio,
                step_list,
                logprob_inf,
                nisqascore_update,
                rawtext,
            ]
            return kk


        random_index = np.random.choice(np.arange(len(self)),
                                        size=batch_size * 8 * 5)
        print('random_index', random_index)
        samp = [self[i] for i in random_index]

##########same trajectory#############
        return DataLoader(
            samp,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker,
            drop_last=True,
            collate_fn=collate_fn)
