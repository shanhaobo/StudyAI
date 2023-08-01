import torch

from datetime import datetime

from ..MultiNNTrainer import MultiNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

class GANTrainer(MultiNNTrainer):
    def __init__(
            self,
            inGenerator : BaseNNModel,
            inDiscriminator : BaseNNModel,
            inGeneratorEmbeddingDim,
            inLearningRate = 1e-5,
            inLogRootPath = "."
        ) -> None:
        super().__init__({"Generator": inGenerator, "Discriminator" : inDiscriminator}, inLearningRate, inLogRootPath)
        
        self.Generator                  = self.NNModuleDict["Generator"]
        self.Discriminator              = self.NNModuleDict["Discriminator"]

        self.GeneratorEmbeddingDim      = inGeneratorEmbeddingDim

        self.EndBatchTrain.add(self.MyEndBatchTrain)

###########################################################################################

    def _CreateOptimizer(self) -> None:
        self.Generator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.Discriminator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        self.Generator.ApplyLossFunc(torch.nn.BCELoss().to(self.Device))
        self.Discriminator.ApplyLossFunc(torch.nn.BCELoss().to(self.Device))

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :

        BatchSize, _, ImageHeight, ImageWidth = inBatchData.size()

        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)
        
        with self.Discriminator as D:
            with self.Generator as G: 
                FakeData = G(
                    torch.randn(
                        (BatchSize, self.GeneratorEmbeddingDim, ImageHeight, ImageWidth),
                        device=self.Device
                    )
                )

                DiscriminatorScores = D(FakeData)
                RealLabels = torch.ones(DiscriminatorScores.size(), device=self.Device)
                
                G.CalcAndAcceptLoss(DiscriminatorScores, RealLabels)

            DiscriminatorScores = D(RealData)
            DLossReal = D.CalcLoss(DiscriminatorScores, RealLabels)

            DiscriminatorScores = D(FakeData.detach())
            FakeLabels = torch.zeros(DiscriminatorScores.size(), device=self.Device)
            DLossFake = D.CalcLoss(DiscriminatorScores, FakeLabels)

            D.AcceptLoss((DLossReal + DLossFake) * 0.5)

###########################################################################################

    def MyEndBatchTrain(self, inArgs, inKVArgs) -> None:
        DLoss, DAvgLoss = self.Discriminator.GetLossValue()
        GLoss, GAvgLoss = self.Generator.GetLossValue()
        print(
            "{} | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | DLoss: {:.6f} / {:.6f} | GLoss: {:.6f} / {:.6f}".
            format(
                datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]"),
                self.CurrEpochIndex,
                self.EndEpochIndex,
                self.CurrBatchIndex + 1,
                self.BatchNumPerEpoch,
                DLoss, DAvgLoss,
                GLoss, GAvgLoss
            )
        )

###########################################################################################
