import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import lightning as L
import pickle as pkl
import os
import shap
import argparse
from models.common import RevIN, ShapProbWrapper, ChebyshevQuantile, ThresholdWeightedCRPS, chebyshev_lobatto_u
from einops import rearrange
from torch.distributions import Normal, StudentT
from neuralforecast.losses.pytorch import DistributionLoss, sCRPS
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.losses import pinball_loss, pinball_loss_expectile
import tempfile
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt


class Model(nn.Module):

    def __init__(self, seq_len, pred_len, d_model, d_ff, dropout, e_layers, activation, embed, freq, n_heads, factor, enc_in,
                method = 'forecast', forecast_task = 'quantile', dist_side = 'both',
                affine = True, scaler= 'revin',
                n_cheb = 2, twcrps_threshold_low = -10.0, twcrps_threshold_high = 10.0, twcrps_side = 'below', twcrps_smooth_h = 0.001, u_grid_size = 256
                ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.method = method
        self.n_cheb = n_cheb
        self.enc_in = enc_in
        self.revin = RevIN(self.enc_in, affine = affine, mode=scaler)
        self.forecast_task = forecast_task
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq,
                                                    dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        if dist_side == 'both' and forecast_task in ['quantile', 'expectile']:
            self.projection = nn.Linear(d_model, self.pred_len * 2)
        elif (dist_side in ['up', 'down'] and forecast_task in ['quantile', 'expectile']) or forecast_task == 'point':
            self.projection = nn.Linear(d_model, self.pred_len) 
        elif forecast_task == 'distribution':
            self.projection = nn.Linear(d_model, self.pred_len * (2 + self.n_cheb))
        self.classify = nn.Linear(d_model, 1)


    def forecast(self, x_enc):

        _, _, N = x_enc.shape
        
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projection(enc_out).permute(0, 2, 1)
        dec_out = self.revin(dec_out, 'denorm')[:, :, - 1]
        dec_out = dec_out.view(dec_out.size(0), self.pred_len, -1)
        return dec_out
    
    def earlywarning(self, x_enc):
        _, _, N = x_enc.shape
        
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        class_out = torch.sigmoid(self.classify(enc_out[:, -1, :]))
        return class_out

    def forward(self, x_enc):
        if self.method == 'forecast':
            dec_out = self.forecast(x_enc)
            if self.forecast_task == 'distribution':
                return dec_out
            else:
                return torch.squeeze(dec_out, dim=-1)
                     
        if self.method == 'earlywarning':
            class_out = self.earlywarning(x_enc)
            class_out = torch.squeeze(class_out, dim=-1)
            return class_out
        

class iTransformer_forecast(L.LightningModule):
    def __init__(self,
                seq_len, pred_len, batch_size, test_batch_size,
                learning_rate, method,
                embed, freq, dropout, e_layers,
                d_model, factor, n_heads, d_ff, activation, enc_in,
                affine, scaler, forecast_task, dist_side, tau_pinball,
                n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h, u_grid_size,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.model = Model(seq_len, pred_len, d_model, d_ff, dropout, e_layers, 
                           activation, embed, freq, n_heads, factor, enc_in, method, 
                           forecast_task, dist_side, affine, scaler, 
                           n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h, u_grid_size
                           )
        self.forecast_task = forecast_task
        if self.forecast_task == 'distribution':
            u = chebyshev_lobatto_u(u_grid_size)
            self.quantile = ChebyshevQuantile(K=n_cheb, u_grid=u, normalize=True)
            self.criterion = ThresholdWeightedCRPS(
                quantile=self.quantile,
                threshold_low=twcrps_threshold_low,
                threshold_high=twcrps_threshold_high,
                side=twcrps_side,
                smooth_h=twcrps_smooth_h
            )
        else:
            self.criterion= self.get_criterion(method, forecast_task, dist_side, tau_pinball)
        self.method = method
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch    
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
            
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        
        loss = self.criterion(outputs, batch_y)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        test_loss = self.criterion(outputs, batch_y)
        self.log('test_loss', test_loss)

        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.y_pred = outputs.detach().cpu()
        else:
            self.x_test =  torch.cat((self.x_test,batch_x.detach().cpu()), dim=0)
            self.y_test =  torch.cat((self.y_test,batch_y.detach().cpu()), dim=0)
            self.y_pred = torch.cat((self.y_pred,outputs.detach().cpu()), dim=0)
    
    def on_test_epoch_end(self):
        '''
        Saves predictions in mlflow artifacts
        '''
        A = {'true' : np.array(self.y_test), 'pred' : np.array(self.y_pred), 'seq' : np.array(self.x_test)}
        pkl.dump(A, open(f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl', 'wb'))
        self.logger.experiment.log_artifact(self.logger.run_id, f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl')

    def predict_step(self, batch, batch_idx):
        return self.model(batch)
    
    def forward(self, batch, batch_idx):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def get_criterion(self, method, forecast_task, dist_side, tau_pinball):
        if forecast_task == 'point':
            criterion = nn.MSELoss()
        elif forecast_task == 'quantile':
            criterion = lambda pred, target: pinball_loss(pred, target, tau_pinball, dist_side)
        elif forecast_task == 'expectile':
            criterion = lambda pred, target: pinball_loss_expectile(pred, target, tau_pinball, dist_side)
        else:
            raise ValueError("distribution criterion is created in __init__")

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('iTransformer')
        # Embedding
        model_parser.add_argument('--embed', type=str, default='fixed')
        model_parser.add_argument('--freq', type=str, default='h')

        
        # Properties of Attn layer
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--factor', type=int, default=3)
        model_parser.add_argument('--n_heads', type=int, default=3)
        model_parser.add_argument('--d_ff', type=int, default=1024)
        model_parser.add_argument('--activation', type=str, default='gelu')

        model_parser.add_argument('--forecast_task', type=str, default = 'quantile', choices=['quantile', 'point', 'expectile', 'distribution'], help='quantile, expectile or point forecasting')
        model_parser.add_argument('--dist_side', type=str, default='both', choices=['both', 'up', 'down'], help='side of the distribution to be predicted (for quantile/expectile forecasting)')
        model_parser.add_argument('--tau_pinball', type=float, help='tau parameter for pinball loss (quantile/expectile regression)', default=0.05)
        model_parser.add_argument('--n_cheb', type=int, default=2, help='number of Chebyshev polynomials for distribution forecasting')

        model_parser.add_argument('--twcrps_threshold_low', type=float, default=-10.0)  # for price target
        model_parser.add_argument('--twcrps_threshold_high', type=float, default=10.0)  # for price target
        model_parser.add_argument('--twcrps_side', type=str, default='two_sided', choices=['below','above', 'two_sided'])
        model_parser.add_argument('--twcrps_smooth_h', type=float, default=2)
        model_parser.add_argument('--u_grid_size', type=int, default=256)

        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--e_layers', type=int, default=2)

        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)

        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        return parent_parser

class iTransformer_classifier(L.LightningModule):
    def __init__(self,
                seq_len, pred_len, batch_size, test_batch_size,
                learning_rate,
                embed, freq, dropout, e_layers,
                d_model, factor, n_heads, d_ff, activation, enc_in,
                affine, scaler, class_loss,
                compute_shap, shap_background_size, shap_test_samples,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.model = Model(seq_len, pred_len, d_model, d_ff, dropout, e_layers, activation, embed, 
                           freq, n_heads, factor, enc_in, method = 'earlywarning', 
                           affine = affine, scaler = scaler, forecast_task = None, dist_side = None)
        self.class_loss = class_loss
        self.threshold = 0.5
        self.val_probs, self.val_true = [], []
        self.test_probs, self.test_true, self.test_seq = [], [], []
        self.criterion= self.get_criterion(class_loss)
        self.compute_shap = compute_shap
        self.shap_background_size = shap_background_size
        self.shap_test_samples = shap_test_samples
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch    
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_probs, self.val_true = [], []

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()  

        outputs = self.model(batch_x)                  
        prob, y = self._ensure_1d_prob_and_target(outputs, batch_y)

        loss = self.criterion(prob, y)

        self.val_probs.append(prob.detach().cpu())
        self.val_true.append(y.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_true) == 0:
            return

        y_true = torch.cat(self.val_true).numpy()
        y_prob = torch.cat(self.val_probs).numpy()

        auc, auprc = self._safe_auc_auprc(y_true, y_prob)
        self.log("val_auc", auc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auprc", auprc, prog_bar=True, on_step=False, on_epoch=True)

        y_pred = (y_prob >= self.threshold).astype(int)

        with tempfile.TemporaryDirectory() as td:
            cm_path = os.path.join(td, "val_confusion_matrix.png")
            self._plot_confusion_matrix(
                y_true=y_true.astype(int),
                y_pred=y_pred,
                title=f"Val Confusion Matrix (thr={self.threshold:.2f})",
                out_path=cm_path
            )
            self._mlflow_log_artifact(cm_path, artifact_path="validation")

    def _shap_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x.float())  # already probabilities
        if out.ndim == 1:
            out = out.unsqueeze(1)
        elif out.ndim == 2 and out.shape[1] != 1:
            # if ever multi-class, you'd handle differently
            out = out[:, :1]
        return out
    
    def _log_shap_on_test(self):
        if not self.trainer.is_global_zero:
            return
        if len(getattr(self, "test_seq", [])) == 0:
            return

        X_test = torch.cat(self.test_seq, dim=0)  # (N, seq_len, n_features)
        n_test = X_test.shape[0]

        n_eval = min(self.shap_test_samples, n_test)
        n_bg   = min(self.shap_background_size, n_test)

        idx = torch.randperm(n_test)[:n_eval]
        bg_idx = torch.randperm(n_test)[:n_bg]

        X_eval = X_test[idx]
        X_bg   = X_test[bg_idx]

        device = next(self.model.parameters()).device
        X_eval = X_eval.to(device)
        X_bg   = X_bg.to(device)

        shap_model = ShapProbWrapper(self.model).to(device)
        shap_model.eval()

        # IMPORTANT: ensure autograd works even if Lightning uses inference_mode=True
        with torch.inference_mode(False), torch.enable_grad():
            explainer = shap.GradientExplainer(shap_model, X_bg)
            shap_values = explainer.shap_values(X_eval)

        # shap_values can be either an array or a list of arrays (one per output)
        if isinstance(shap_values, list):
            shap_arr = shap_values[0]
        else:
            shap_arr = shap_values

        X_eval_cpu = X_eval.detach().cpu().numpy()
        idx_cpu = idx.detach().cpu().numpy()

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            npz_path = td / "shap_test_subset.npz"
            np.savez_compressed(
                npz_path,
                shap_values=shap_arr,   # typically (B, seq_len, n_features)
                x_eval=X_eval_cpu,
                test_indices=idx_cpu
            )
            self._mlflow_log_artifact(str(npz_path), artifact_path="test/shap")

            # Optional: summary plot (aggregate over time -> (B, n_features))
            shap_agg = np.abs(shap_arr).mean(axis=1)  # mean over seq_len
            x_agg = X_eval_cpu.mean(axis=1)

            plt.figure(figsize=(8, 4), dpi=150)
            shap.summary_plot(shap_agg, features=x_agg, show=False)
            fig_path = td / "shap_summary_mean_over_time.png"
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            self._mlflow_log_artifact(str(fig_path), artifact_path="test/shap")

    def on_test_epoch_start(self):
        self.test_probs, self.test_true, self.test_seq = [], [], []

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)                  # probs
        prob, y = self._ensure_1d_prob_and_target(outputs, batch_y)

        loss = self.criterion(prob, y)

        self.test_probs.append(prob.detach().cpu())
        self.test_true.append(y.detach().cpu())
        self.test_seq.append(batch_x.detach().cpu())   # optional, can remove if large

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        if len(self.test_true) == 0:
            return

        y_true = torch.cat(self.test_true).numpy()
        y_prob = torch.cat(self.test_probs).numpy()
        x_seq = torch.cat(self.test_seq, dim=0).numpy()

        auc, auprc = self._safe_auc_auprc(y_true, y_prob)
        self.log("test_auc", auc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_auprc", auprc, prog_bar=True, on_step=False, on_epoch=True)

        y_pred = (y_prob >= self.threshold).astype(int)

        payload = {
            "true": y_true,
            "prob": y_prob,
            "pred": y_pred,
            "seq": x_seq,  # remove if too big
            "threshold": self.threshold,
        }

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            pkl_path = td / "preds_test_set.pkl"
            with open(pkl_path, "wb") as f:
                pkl.dump(payload, f)

            cm_path = td / "test_confusion_matrix.png"
            self._plot_confusion_matrix(
                y_true=y_true.astype(int),
                y_pred=y_pred,
                title=f"Test Confusion Matrix (thr={self.threshold:.2f})",
                out_path=str(cm_path)
            )

            self._mlflow_log_artifact(str(pkl_path), artifact_path="test")
            self._mlflow_log_artifact(str(cm_path), artifact_path="test")
        if self.compute_shap:
            try:
                self._log_shap_on_test()
            except Exception as e:
                print(f"----- Error computing SHAP values on test set: {e} -----")
    
    def predict_step(self, batch, batch_idx):
        return self.model(batch)
    
    def forward(self, batch, batch_idx):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def get_criterion(self, class_loss):
        if class_loss == 'bce':
            criterion = nn.BCELoss()
        return criterion
    
    def _ensure_1d_prob_and_target(self, outputs: torch.Tensor, batch_y: torch.Tensor):
        """
        outputs: probs from model, expected shape (B,) or (B,1)
        batch_y: shape (B,)
        returns: (prob_1d, y_1d) both float tensors on same device
        """
        prob = outputs
        if prob.ndim > 1:
            prob = prob.squeeze(-1)
        prob = prob.float().clamp(0.0, 1.0)

        y = batch_y
        if y.ndim > 1:
            y = y.squeeze(-1)
        y = y.float()
        return prob, y


    def _safe_auc_auprc(self, y_true: np.ndarray, y_prob: np.ndarray):
        # AUC/AUPRC undefined if only one class present
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        try:
            auprc = average_precision_score(y_true, y_prob)
        except Exception:
            auprc = np.nan
        return auc, auprc


    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["0", "1"], yticklabels=["0", "1"],
            xlabel="Predicted", ylabel="True", title=title
        )

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


    def _mlflow_log_artifact(self, local_path: str, artifact_path: str):
        # only log once in DDP
        if not self.trainer.is_global_zero:
            return
        self.logger.experiment.log_artifact(self.logger.run_id, local_path, artifact_path=artifact_path)

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('iTransformer')
        # Embedding
        model_parser.add_argument('--embed', type=str, default='fixed')
        model_parser.add_argument('--freq', type=str, default='h')

        
        # Properties of Attn layer
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--factor', type=int, default=3)
        model_parser.add_argument('--n_heads', type=int, default=3)
        model_parser.add_argument('--d_ff', type=int, default=1024)
        model_parser.add_argument('--activation', type=str, default='gelu')

        model_parser.add_argument('--class_loss', type=str, default='bce', choices=['bce', 'focal'], help='loss function for classification task')

        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--e_layers', type=int, default=2)

        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--learning_rate', type=float,default=0.0001)

        model_parser.add_argument('--compute_shap', type=int, choices=[0,1], default=0, help='whether to compute SHAP values at test time')
        model_parser.add_argument('--shap_background_size', type=int, default=64, help='number of background samples for SHAP')
        model_parser.add_argument('--shap_test_samples', type=int, default=256, help='number of test samples to compute SHAP values for (keep small; SHAP can be expensive)')

        return parent_parser

