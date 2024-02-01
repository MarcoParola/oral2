import torch
import torchvision
from pytorch_lightning import LightningModule
import tensorboard as tb

class ImageVAE(LightningModule):
    def __init__(self, latent_dim=64, lr = 0.0004, max_epochs = 100):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Dropout(0.15)
        )
        self.maxpooling_3_2 = torch.nn.MaxPool2d(3, stride = 2, return_indices=True)

        self.encoder_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(0.15)
        )

        self.encoder_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.15)
        )

        self.maxpooling_2_2 = torch.nn.MaxPool2d(2, stride = 2, return_indices=True)

        self.encoder_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Dropout(0.15),

            torch.nn.Flatten(),
            #torch.nn.ReLU(),
        )

        self.layer1 = torch.nn.Linear(512 * 2 * 2, latent_dim)#128,64
        self.layer2 = torch.nn.Linear(512 * 2 * 2, latent_dim)

        # Decoder
        self.decoder_block1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512 * 2 * 2),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (512,2,2)),

            torch.nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.15),
            #torch.nn.Upsample(scale_factor=2)
        )
        self.maxunpool_2_2 = torch.nn.MaxUnpool2d(2, stride=2, padding=0)

        self.decoder_block2 = torch.nn.Sequential(        
            torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(0.15)
        )
        
        self.maxunpool_3_2 = torch.nn.MaxUnpool2d(3, stride=2, padding=0)

        self.decoder_block3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Dropout(0.15),
        )

        self.decoder_block4 = torch.nn.Sequential(     
            torch.nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        #x = self.encoder(x)
        encoded1 = self.encoder_block1(x)
        #print(encoded1.shape)
        encodedM1, index1 = self.maxpooling_3_2(encoded1)
        #print(encodedM1.shape)
        encoded2 = self.encoder_block2(encodedM1)
        #print(encoded2.shape)
        encodedM2, index2 = self.maxpooling_3_2(encoded2)
        #print(encodedM2.shape)
        encoded3 = self.encoder_block3(encodedM2)
        #print(encoded3.shape)
        encodedM3, index3 = self.maxpooling_2_2(encoded3)
        #print(encodedM3.shape)
        x = self.encoder_block4(encodedM3)

        mu =  self.layer1(x)
        sigma = torch.exp(self.layer2(x))
        x = mu + sigma*self.N.sample(mu.shape)
        #self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        #print(x.shape)
        #x = self.decoder(x) 

        decoded = self.decoder_block1(x)
        decoded = self.maxunpool_2_2(decoded, index3, output_size=encoded3.size())
        #print(decoded.shape)
        decoded = self.decoder_block2(decoded)
        #print(decoded.shape)
        decoded = self.maxunpool_3_2(decoded, index2, output_size=encoded2.size())
        #print(decoded.shape)
        decoded = self.decoder_block3(decoded)
        #print(decoded.shape)
        decoded = self.maxunpool_3_2(decoded, index1, output_size=encoded1.size())
        #print(decoded.shape)
        x = self.decoder_block4(decoded)
        #print(x.shape)

        #print(x.shape)
        #x = x.reshape((-1, 1, 768, 768))
        return x, mu, sigma

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train") 

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val") 
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")
        #Calcola solo mse

    def _common_step(self, batch, batch_idx, stage):
        x, _ = batch
        reconstruction, mu, sigma = self.forward(x)
        
        reconstruction_loss = self.loss(reconstruction, x)
        
        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        loss = reconstruction_loss + kl_loss

        self.log(f"{stage}_recon_loss", reconstruction_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-5)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        #use SGD

