from pytorch_lightning import LightningDataModule
from pathlib import Path
from buffer_jc import RolloutBuffer
import dataloader


class DataModule(LightningDataModule):

    def __init__(self, buffer_path: Path, hparams_rl):
        super().__init__()
        save_dir = Path(
            "/users/PAS2062/delijingyic/project/MOSNet/output/CNN-BLSTM_mse_16/modelPara"
        )
        if (save_dir.exists()):
            model_list = list(save_dir.iterdir())
        else:
            model_list = []

        if len(model_list) > 0:
            last_model = model_list[-1]
        self.buffer = RolloutBuffer(buffer_path, )
        self.hparams_rl = hparams_rl

    def prepare_data(self, ):
        self.buffer.load_data()

    def setup(self, stage: str):
        self.buffer.load_data()

    def train_dataloader(self):
        print('call_train_dataloader')
        return self.buffer.create_dataloader(2, 2)

    def val_dataloader(self):

        return dataloader.create_dataloader(self.hparams_rl, 0)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...
