import pytorch_lightning as pl
import torch
from torch import nn

class Classifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3, optimizer=torch.optim.Adam, optimizer_params={}, scheduler=None,
                 scheduler_params={}, monitor_value=None, loss_function=nn.CrossEntropyLoss):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.monitor_value = monitor_value
        self.loss_function = loss_function

    @staticmethod
    def binary_accuracy(output, batch_y):
        prediction = output // 0.5
        idcs_0 = torch.nonzero(batch_y == 0, as_tuple=True)
        total_0 = prediction[idcs_0].numel()
        # count_nonzero not available in pytorch 1.6.0
        correct_0 = int(torch.where(prediction[idcs_0] == 0, torch.ones_like(prediction[idcs_0]),
                                    torch.zeros_like(prediction[idcs_0])).sum())
        idcs_1 = torch.nonzero(batch_y == 1, as_tuple=True)
        total_1 = prediction[idcs_1].numel()
        # count_nonzero not available in pytorch 1.6.0
        correct_1 = int(torch.where(prediction[idcs_1] == 1, torch.ones_like(prediction[idcs_1]),
                                    torch.zeros_like(prediction[idcs_1])).sum())

        return {'correct_0': correct_0, 'total_0': total_0, 'correct_1': correct_1, 'total_1': total_1}

    def training_step(self, batch, batch_index):
        '''
        This function takes a batch of samples and batch_index as input.
        The batch is split into its X (input) and y (output) values,
        the X will be fed to the network and its output is stored as 'output'.
        Then the loss function will be applied on the output and y, and the loss and accuracy will be logged.
        This function may not call 'validation_step' even though both functions perform similar operations.
        'validation_step' is wrapped with a 'torch.set_grad_enabled(False)', so gradients will not be calculated!
        '''
        batch_x, batch_y = batch
        output = self.model.forward(batch_x)  # Forward pass
        loss = self.loss_function(output, batch_y)  # Calculate loss
        bin_acc = self.binary_accuracy(output, batch_y)
        self.log_dict({'Training loss': loss, 'correct_0': bin_acc['correct_0'], 'total_0': bin_acc['total_0'],
                       'correct_1': bin_acc['correct_1'], 'total_1': bin_acc['total_1']}, logger=True, on_step=True)
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        '''
        Training loss and accuracy is already monitored at each step, so we won't do anything specifically after a training epoch.
        '''
        pass

    def validation_step(self, batch, batch_index):
        '''
        This function operates in a similar way as 'training_step', it is wrapped in a 'torch.set_grad_enabled(False)', in order to prevent gradient calcualtion and update.
        The only difference is that we won't log the validation loss and accuracy every step, as we only want this after every validation epoch
        '''
        batch_x, batch_y = batch
        output = self.model.forward(batch_x)
        loss = self.loss_function(output, batch_y)  # Calculate loss
        bin_acc = self.binary_accuracy(output, batch_y)
        return {"loss": loss, 'correct_0': bin_acc['correct_0'], 'total_0': bin_acc['total_0'],
                'correct_1': bin_acc['correct_1'], 'total_1': bin_acc['total_1']}

    def validation_epoch_end(self, validation_step_outputs):
        # log the accuracy and loss after one epoch
        self.log_dict(self.end_epoch(validation_step_outputs, desc='Validation'), logger=True)

    def test_step(self, batch, batch_index):
        # At each batch testing step we'll calculate the loss
        return self.validation_step(batch, batch_index)

    def test_epoch_end(self, test_step_outputs):
        # log the accuracy and loss after one test epoch (so usually only once)
        self.log_dict(self.end_epoch(test_step_outputs, desc='Test'), logger=True)

    def end_epoch(self, step_outputs, desc=''):
        # Calculate the accuracy and loss of an epoch
        loss = round(torch.stack([step['loss'] for step in step_outputs]).mean().item(), 3)
        correct_0 = sum([step["correct_0"] for step in step_outputs])
        total_0 = sum([step["total_0"] for step in step_outputs])
        correct_1 = sum([step["correct_1"] for step in step_outputs])
        total_1 = sum([step["total_1"] for step in step_outputs])
        return {f'{desc} loss': loss, f'{desc} accuracy 0': round(correct_0 / total_0, 3),
                f'{desc} correct 1': correct_1, f'{desc} total 1': total_1}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        if self.scheduler:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor_value}
        return {"optimizer": optimizer}
