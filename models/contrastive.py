import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning.core.module import LightningModule
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LinearLR
from lightning.pytorch.utilities import grad_norm
from models.cgcnn import CGCNN
from losses.clip_loss import clip_loss, cosface_loss, SigLIPLoss
from losses.utils import calc_grobal_top_k_acc, batch_wise_accuracy
from models.text_encoder import HuggingFaceEncoder
from models.utils import normalize_embedding
# from eval_scripts.zero_shot_evaluation import calculate_top_k_accuracy, calculate_similarities


class ClaspModel(LightningModule):
    """A class representing a metric learning model using LightningModule.

    Attributes:
    cfg: Model configs such as learning rate, encoder name etc.
    """
    def __init__(self, cfg):
        super(ClaspModel, self).__init__()

        if cfg.encoder_name == "cgcnn":
            self.model = CGCNN(cfg)
        else:
            raise Exception(f"Invalid cfg.encoder_name: {cfg.encoder_name}")

        self.model_text = HuggingFaceEncoder(cfg)
        if cfg.loss_fn == 'clip_loss':
            self.loss_fn = clip_loss
        elif cfg.loss_fn == 'cosface_loss':
            self.loss_fn = cosface_loss
        elif cfg.loss_fn == 'siglip_loss':
            self.loss_fn = SigLIPLoss(cfg)
        else:
            raise ValueError(f"Invalid cfg.loss_fn: {cfg.loss_fn}")

        self.cfg = cfg
        # self.training_step_outputs = []
        self.validation_step_outputs = []


    # def load_state_dict(self, state_dict, strict: bool = True):
    #     # Override for backward compatibility
    #     new_dict = {}
    #     for key in state_dict:
    #         if key.startswith("model.xrd_"):
    #             # replace 'model' with 'model_text'
    #             new_dict['model_text' + key[5:]] = state_dict[key]
    #         else:
    #             new_dict[key] = state_dict[key]

    #     return super().load_state_dict(new_dict, strict)

    def training_step(self, batch, batch_idx):
        output_cry = self.model(batch)
        output_text = self.model_text(batch)
        if self.cfg.embedding_normalize is not None:
            output_text = normalize_embedding(output_text, self.cfg.embedding_normalize)
            output_cry = normalize_embedding(output_cry, self.cfg.embedding_normalize)
        # print(f"output_cry shape (before gather): {output_cry.shape}")
        output_cry = self.all_gather(output_cry, sync_grads=True)
        # print(f"output_cry shape (afrer gather): {output_cry.shape}")
        output_text = self.all_gather(output_text, sync_grads=True)
        num_elements = output_cry.shape[0] * output_cry.shape[1]
        output_cry = output_cry.view(num_elements, *output_cry.shape[2:])
        # print(f"output_cry shape (afrer reshape): {output_cry.shape}")
        output_text = output_text.view(num_elements, *output_text.shape[2:])
        # Check if self.loss_fn is an nn.Module instance or a function
        if isinstance(self.loss_fn, nn.Module):
            # If instance, call without cfg
            loss = self.loss_fn(output_text, output_cry)
        else:
            # If function, call with cfg as parameter
            loss = self.loss_fn(output_text, output_cry, self.cfg)

        _, batch_acc= batch_wise_accuracy(output_text, output_cry)

        output = {
            'loss': loss,
            'progress_bar': {'tr/loss': loss, 'tr/acc':batch_acc},
            'log': {'train/loss': loss, 'train/batch_acc':batch_acc}
        }
        self.log('train/loss', loss.to("cuda"), sync_dist=True)
        self.log('train/batch_acc', batch_acc.to("cuda"), sync_dist=True)
        if isinstance(self.loss_fn, SigLIPLoss):
            t = torch.exp(self.loss_fn.logit_scale)
            b = self.loss_fn.logit_bias
            self.log('train/siglip_t', t.to("cuda"), sync_dist=True)
            self.log('train/siglip_b', b.to("cuda"), sync_dist=True)
            # logits = self.loss_fn.logits
            # self.logger.experiment.add_histogram("train/siglip/logits", logits.detach().cpu().float().numpy(), self.global_step)
        return output

    def validation_step(self, batch, batch_idx):
        output_cry = self.model(batch)
        output_text = self.model_text(batch)
        if self.cfg.embedding_normalize is not None:
            output_cry = normalize_embedding(output_cry, self.cfg.embedding_normalize)
            output_text = normalize_embedding(output_text, self.cfg.embedding_normalize)
        output_cry = self.all_gather(output_cry, sync_grads=True)
        output_text = self.all_gather(output_text, sync_grads=True)
        num_elements = output_cry.shape[0] * output_cry.shape[1]
        output_cry = output_cry.view(num_elements, *output_cry.shape[2:])
        output_text = output_text.view(num_elements, *output_text.shape[2:])

        if isinstance(self.loss_fn, nn.Module):
            loss = self.loss_fn(output_text, output_cry)
        else:
            loss = self.loss_fn(output_text, output_cry, self.cfg)
        _, batch_acc = batch_wise_accuracy(output_text, output_cry)
        output = {'val/loss': loss, 
                  'val/acc': batch_acc.float(),
                  'out_cry': output_cry.detach().cpu(),
                  'out_text': output_text.detach().cpu(),
                 }
        self.validation_step_outputs.append(output)
        
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        acc = torch.stack([x['val/acc'] for x in outputs]).mean()
        cry = torch.cat([x['out_cry'] for x in outputs], dim=0)
        text = torch.cat([x['out_text'] for x in outputs], dim=0)
        topk_acc = calc_grobal_top_k_acc(embedding_query=text, embedding_target=cry, k=10)
        logs = {'val/loss': avg_loss, 'val/acc': acc}
        
        self.log('val/loss', avg_loss.to("cuda"), sync_dist=True)
        self.log('val/acc', acc.to("cuda"), sync_dist=True)
        for i in range(len(topk_acc)):
            logs['val/top%02d' % (i+1)] = torch.tensor(topk_acc[i])
            self.log('val/top%02d' % (i+1), torch.tensor(topk_acc[i]).to("cuda"), sync_dist=True)
        
        if self.global_rank == 0:
            print("######")
            print(f"val loss: {avg_loss:.3f}")
            print(f'val acc: {acc*100:4.2f} ')
            print("evaluating text->crystal serch acc...")
            for i in range(len(topk_acc)):
                print(f'top{i+1}: {topk_acc[i]*100:4.2f} ')
            print("######")
        
        self.validation_step_outputs = []
        return {'log': logs}

    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)
        
    # def test_epoch_end(self, outputs):
    #     out = self.validation_epoch_end(outputs)
    #     val_log = out['log']
    #     test_log = {}
    #     for key in val_log:
    #         newkey = key.replace('val/', 'test/')
    #         test_log[newkey] = val_log[key]
    #         self.log(newkey, val_log[key])
    #     return {'log': test_log}

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training.

        Selects the optimizer class (Adam, SGD, or AdamW) based on the configuration.
        Collects parameters from both the main model and the text model.
        If the loss function has learnable parameters (i.e., is an nn.Module), those are also included.
        Raises:
            ValueError: If the specified optimizer is not recognized.

        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        opt_class = {
            'Adam': Adam,
            'SGD': SGD,
            'AdamW': AdamW
        }
        optimizer_name = getattr(self.cfg, 'optimizer', 'Adam')
        if optimizer_name not in opt_class:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized.")

        params_to_optimize = list(self.model.parameters()) + list(self.model_text.parameters())
        if isinstance(self.loss_fn, nn.Module):
            params_to_optimize += list(self.loss_fn.parameters())

        optimizer = opt_class[optimizer_name](params_to_optimize, lr=self.cfg.lr)
        return optimizer


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
        norms = grad_norm(self.model_text, norm_type=2)
        self.log_dict(norms)

    def forward(self, x):
        output_cry = self.model(x)
        output_text = self.model_text(x)
        if self.cfg.embedding_normalize is not None:
            output_text = normalize_embedding(output_text, self.cfg.embedding_normalize)
            output_cry = normalize_embedding(output_cry, self.cfg.embedding_normalize)
        
        return output_cry, output_text
