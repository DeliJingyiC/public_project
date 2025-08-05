import torch
from typing import List
import torch.nn.functional as F
from model.base import BaseModule
import numpy as np
from torch.distributions.normal import Normal


class Window(BaseModule):

    def __init__(self, hparams):
        super(Window, self).__init__()
        self.hparams = hparams
        self.scale = hparams.window.scale
        self.length = hparams.window.length

    def forward_test(
        self,
        y_clean: torch.Tensor,
        hidden_rep: torch.Tensor,
        output_lengths,
        y_sync: torch.Tensor,
    ):
        # print(y_clean.shape, "y_clean")
        # print(output_lengths, "output_lengths")
        # print(self.scale, "self.scale")
        # print(self.length, "self.length")

        assert y_clean.shape == y_sync.shape, f"{y_clean.shape}, {y_sync.shape}"
        y_clean_sliced = []
        hidden_rep_sliced = []
        y_sync_sliced = []

        sample_length = int(self.length * self.scale)  #76800
        print('y_clean', y_clean.size())
        for i in range(y_clean.size(0)):

            # assert False, (y_clean[i].shape[0] / self.scale).dtype
            print('hidden_rep[i].shape', len(hidden_rep[i]))
            print('hidden_rep[i].shape', hidden_rep[i].shape)

            if (len(y_clean[i]) <= sample_length) or (len(hidden_rep[i])
                                                      <= self.length):
                # kkkk = torch.zeros((sample_length, *y_clean[i].shape[1:]))
                # kbound = min(sample_length, len(y_clean[i]))
                # kkkk[:kbound] = y_clean[i, :kbound]
                # y_clean_sliced.append(kkkk)
                # if len(hidden_rep[i]) > self.length:
                #     continue

                y_clean_sliced.append(
                    F.pad(
                        y_clean[i],
                        (0, sample_length - len(y_clean[i])),
                        mode='constant',
                        value=0.0,
                    ))
                print('y_clean[i]', y_clean[i].shape)
                print('(0, sample_length - len(y_sync[i]))',
                      (0, sample_length - len(y_sync[i])))

                y_sync_sliced.append(
                    F.pad(
                        y_clean[i],
                        (0, sample_length - len(y_sync[i])),
                        mode='constant',
                        value=0.0,
                    ))
                if self.length - hidden_rep.shape[2] < 0:
                    print('hidden_rep.shape[2]', hidden_rep.shape[2])
                    print('hidden_rep[i]', hidden_rep[i].shape)
                    start_index = np.random.randint(
                        0,
                        hidden_rep.shape[2] - self.length,
                    )

                    hidden_rep_sliced.append(
                        hidden_rep[i, :,
                                   start_index:start_index + self.length])
                else:
                    hidden_rep_sliced.append(
                        F.pad(
                            hidden_rep[i],
                            # (0, self.length - len(hidden_rep[i])),
                            (0, self.length - hidden_rep.shape[2]),
                            mode='constant',
                            value=0.0,
                        ))
                # hbound = min(self.length, hidden_rep[i].shape[-1])
                # hhhh = torch.zeros((*hidden_rep[i].shape[:-1], self.length))
                # hhhh[:, :hbound] = hidden_rep[i, :, :hbound]
                # hidden_rep_sliced.append(hidden_rep[i, :, 0:0 + self.length])

                continue

            # min_startint = min(output_lengths[i].item() - self.length,
            #                    (len(y_clean[i]) - sample_length) / self.scale)
            min_startint = min(hidden_rep.shape[2] - self.length,
                               (len(y_clean[i]) - sample_length) / self.scale)
            print('hidden_rep[i]) <= self.length', hidden_rep[i].shape)
            print('hidden_rep.shape[2] - self.length', hidden_rep.shape[2])

            # if output_lengths[i].item() - self.length>0:
            #     start_index = np.random.randint(
            #         0,
            #         hidden_rep.shape[2] - self.length,
            #     )
            # else:
            #     start_index = np.random.randint(
            #         0,
            #             (len(y_clean[i]) - sample_length) / self.scale,
            #     )
            hidden_rep_boundary = hidden_rep.shape[2] - self.length
            y_clean_boundary = (len(y_clean[i]) - sample_length) / self.scale
            if hidden_rep_boundary > 0 and y_clean_boundary > 0:
                start_index = np.random.randint(
                    0, min(hidden_rep_boundary, y_clean_boundary))
            else:
                start_index = 0
            scaled_start_index = self.scale * start_index

            hidden_rep_sliced.append(hidden_rep[i, :, start_index:start_index +
                                                self.length])

            # print('startidx,scale startindx', start_index, scaled_start_index)
            # print('hidden_rep_len', hidden_rep.shape)
            # print('hidden_rep_sliced', [i.shape for i in hidden_rep_sliced])
            print('y_cleanscaled_start_index', scaled_start_index)
            print('scaled_start_index + sample_length',
                  scaled_start_index + sample_length)

            assert y_clean[i, scaled_start_index:scaled_start_index +
                           sample_length].shape[0] == 76800, [
                               y_clean[i, :].shape[0],
                               y_clean[i,
                                       scaled_start_index:scaled_start_index +
                                       sample_length].shape[0],
                               y_clean[i,
                                       scaled_start_index:scaled_start_index +
                                       sample_length].shape,
                               scaled_start_index,
                               scaled_start_index + sample_length
                           ]

            y_clean_sliced.append(
                y_clean[i,
                        scaled_start_index:scaled_start_index + sample_length])
            # for i in y_clean_sliced:
            #     assert i.shape[0] == 76800, [
            #         scaled_start_index, scaled_start_index + sample_length
            #     ]
            print('y_clean_sliced', [i.shape for i in y_clean_sliced])

            y_sync_sliced.append(
                y_sync[i,
                       scaled_start_index:scaled_start_index + sample_length])
        y_clean_sliced = torch.stack(y_clean_sliced, dim=0)
        y_sync_sliced = torch.stack(y_sync_sliced, dim=0)

        hidden_rep_sliced = torch.stack(hidden_rep_sliced, dim=0)
        if hidden_rep_sliced.size(-1) < self.length:
            hidden_rep_sliced = F.pad(
                hidden_rep_sliced,
                (0, self.length - hidden_rep_sliced.size(-1)),
                mode='constant',
                value=0.0,
            )
            y_clean_sliced = F.pad(
                y_clean_sliced,
                (0, sample_length - y_clean_sliced.size(-1)),
                mode='constant',
                value=0.0,
            )
            y_sync_sliced = F.pad(
                y_sync_sliced,
                (0, sample_length - y_clean_sliced.size(-1)),
                mode='constant',
                value=0.0,
            )
        # y_clean_sliced = y_clean_sliced.cuda()
        # hidden_rep_sliced = hidden_rep_sliced.cuda()
        return y_clean_sliced.type_as(y_clean), hidden_rep_sliced.type_as(
            y_clean), y_sync_sliced.type_as(y_clean)

    def forward_i(
        self,
        y_clean,
        hidden_rep,
        output_lengths,
    ):
        y_clean_sliced = list()
        hidden_rep_sliced = list()
        y_sync_sliced = list()
        log_prob_sliced = list()

        # input(y_clean.shape)

        for i in range(y_clean.size(0)):
            if output_lengths[i] > self.length:
                start_index = torch.randint(
                    0,
                    output_lengths[i] - self.length,
                    (1, ),
                ).squeeze(0)
            else:
                start_index = 0
            hidden_rep_sliced.append(
                hidden_rep[i, :, start_index:start_index + self.length], )

            y_clean_sliced_ = y_clean[i, self.scale * start_index:self.scale *
                                      (start_index + self.length)]
            if y_clean_sliced_.shape[0] != self.scale * self.length:
                y_clean_sliced_ = F.pad(
                    y_clean_sliced_,
                    (0, self.scale * self.length - y_clean_sliced_.shape[0]),
                    mode='constant',
                    value=0.0,
                )
            y_clean_sliced.append(y_clean_sliced_)

            y_sync_sliced_ = y_sync[i, self.scale * start_index:self.scale *
                                    (start_index + self.length)]
            if y_sync_sliced_.shape[0] != self.scale * self.length:
                y_sync_sliced_ = F.pad(
                    y_sync_sliced_,
                    (0, self.scale * self.length - y_sync_sliced_.shape[0]),
                    mode='constant',
                    value=0.0,
                )
            y_sync_sliced.append(y_sync_sliced_)

            print('log_prob.shape', log_prob.shape)
            log_prob_sliced_ = log_prob[i,
                                        self.scale * start_index:self.scale *
                                        (start_index + self.length)]
            print('log_prob_sliced_before', log_prob_sliced_.shape)
            print('log_prob_sliced_before[0]', log_prob_sliced_.shape[0])

            if log_prob_sliced_.shape[0] != self.scale * self.length:
                print('enter')
                log_prob_sliced_ = F.pad(
                    log_prob_sliced_,
                    (0, self.scale * self.length - log_prob_sliced_.shape[0]),
                    mode='constant',
                    value=Normal(torch.zeros_like(model_std),
                                 model_std).log_prob(
                                     torch.zeros_like(model_std))[i].item(),
                )
            print('log_prob_sliced_after', log_prob_sliced_.shape)

            log_prob_sliced.append(log_prob_sliced_)

        y_clean_sliced = torch.stack(y_clean_sliced, dim=0)
        y_sync_sliced = torch.stack(y_sync_sliced, dim=0)
        log_prob_sliced = torch.stack(log_prob_sliced, dim=0)

        print(
            'Normal(torch.zeros_like(model_std),model_std).log_prob(torch.zeros_like(model_std))[i]',
            Normal(torch.zeros_like(model_std),
                   model_std).log_prob(torch.zeros_like(model_std))[i].item())
        hidden_rep_sliced = torch.stack(hidden_rep_sliced, dim=0)
        if hidden_rep_sliced.size(-1) < self.length:
            hidden_rep_sliced = F.pad(
                hidden_rep_sliced,
                (0, self.length - hidden_rep_sliced.size(-1)),
                mode='constant',
                value=0.0,
            )
            y_clean_sliced = F.pad(
                y_clean_sliced,
                (0, self.scale * self.length - y_clean_sliced.size(-1)),
                mode='constant',
                value=0.0,
            )
            y_sync_sliced = F.pad(
                y_sync_sliced,
                (0, self.scale * self.length - y_sync_sliced.size(-1)),
                mode='constant',
                value=0.0,
            )
            log_prob_sliced = F.pad(
                log_prob_sliced,
                (0, self.scale * self.length - log_prob_sliced.size(-1)),
                mode='constant',
                value=Normal(torch.zeros_like(model_std), model_std).log_prob(
                    torch.zeros_like(model_std))[i].item(),
            )
        return [
            y_clean_sliced.type_as(y_clean),
            hidden_rep_sliced.type_as(y_clean),
            y_sync_sliced.type_as(y_clean),
            log_prob_sliced.type_as(y_clean),
        ]
    def forward(self, y_clean, hidden_rep, output_lengths):
        y_clean_sliced = list()
        hidden_rep_sliced = list()
        for i in range(y_clean.size(0)):
            if output_lengths[i] > self.length:
                start_index = torch.randint(0, output_lengths[i] - self.length, (1,)).squeeze(0)
            else:
                start_index = 0
            hidden_rep_sliced.append(hidden_rep[i, :, start_index:start_index +
                                            self.length])
            y_clean_sliced.append(y_clean[i, self.scale * start_index:self.scale * (start_index + self.length)])
        y_clean_sliced = torch.stack(y_clean_sliced, dim=0)
        hidden_rep_sliced = torch.stack(hidden_rep_sliced, dim=0)
        if hidden_rep_sliced.size(-1) < self.length:
            hidden_rep_sliced = F.pad(hidden_rep_sliced, (0, self.length - hidden_rep_sliced.size(-1)),
                                      mode='constant', value=0.0)
            y_clean_sliced = F.pad(y_clean_sliced, (0, self.scale * self.length - y_clean_sliced.size(-1)),
                                      mode='constant', value=0.0)
        return y_clean_sliced, hidden_rep_sliced