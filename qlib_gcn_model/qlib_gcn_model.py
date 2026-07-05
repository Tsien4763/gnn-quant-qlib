from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, Optional, Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import get_or_create_path


class GraphConvolution(nn.Module):
    """Simple GCN layer: H' = A_norm H W + b."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, node_feature: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = self.linear(node_feature)
        return torch.matmul(adj_norm, support)


class TemporalGCNNet(nn.Module):
    """RNN encoder + two-layer graph convolution + prediction head."""

    ADJ_MODES = {"dynamic", "full", "identity", "external"}

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        gcn_hidden_size: int = 64,
        dropout: float = 0.1,
        base_model: str = "GRU",
        adj_mode: str = "dynamic",
        topk: int = 20,
        min_edge_weight: float = 0.0,
        add_residual: bool = True,
    ):
        super().__init__()
        if d_feat <= 0:
            raise ValueError("d_feat must be positive.")
        if adj_mode not in self.ADJ_MODES:
            raise ValueError(f"adj_mode must be one of {sorted(self.ADJ_MODES)}.")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.gcn_hidden_size = gcn_hidden_size
        self.dropout_rate = dropout
        self.adj_mode = adj_mode
        self.topk = topk
        self.min_edge_weight = min_edge_weight
        self.add_residual = add_residual

        rnn_dropout = dropout if num_layers > 1 else 0.0
        base_model = base_model.upper()
        if base_model == "GRU":
            self.encoder = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif base_model == "LSTM":
            self.encoder = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError("base_model must be GRU or LSTM.")

        self.gcn1 = GraphConvolution(hidden_size, gcn_hidden_size)
        self.gcn2 = GraphConvolution(gcn_hidden_size, gcn_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        head_input_dim = gcn_hidden_size + hidden_size if add_residual else gcn_hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def normalize_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Symmetric GCN normalization: D^-1/2 (A + I) D^-1/2."""
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be a square 2D tensor, got shape {tuple(adj.shape)}.")
        n_nodes = adj.shape[0]
        eye = torch.eye(n_nodes, dtype=adj.dtype, device=adj.device)
        adj = adj + eye
        degree = adj.sum(dim=1).clamp_min(eps)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        return degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]

    @staticmethod
    def drop_self_edges(adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adj.shape[0], dtype=torch.bool, device=adj.device)
        return adj.masked_fill(eye, 0.0)

    def build_dynamic_adj(self, hidden: torch.Tensor) -> torch.Tensor:
        """Build a sparse daily graph from cosine similarity between node embeddings."""
        n_nodes = hidden.shape[0]
        if n_nodes == 1:
            return torch.ones((1, 1), dtype=hidden.dtype, device=hidden.device)

        normalized = F.normalize(hidden, p=2, dim=-1, eps=1e-12)
        adj = torch.relu(torch.matmul(normalized, normalized.T))
        adj = self.drop_self_edges(adj)

        if self.min_edge_weight > 0:
            adj = adj * (adj >= self.min_edge_weight).to(adj.dtype)

        if self.topk is not None and self.topk > 0 and self.topk < n_nodes:
            # Keep top-k neighbours for each node. Self-similarity is retained and later
            # the adjacency is symmetrized, so information can flow both ways.
            _, indices = torch.topk(adj, k=self.topk, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, indices, 1.0)
            adj = adj * mask

        adj = torch.maximum(adj, adj.T)
        return self.normalize_adj(adj)

    def build_full_adj(self, hidden: torch.Tensor) -> torch.Tensor:
        n_nodes = hidden.shape[0]
        adj = torch.ones((n_nodes, n_nodes), dtype=hidden.dtype, device=hidden.device)
        adj = self.drop_self_edges(adj)
        return self.normalize_adj(adj)

    def build_identity_adj(self, hidden: torch.Tensor) -> torch.Tensor:
        n_nodes = hidden.shape[0]
        return torch.eye(n_nodes, dtype=hidden.dtype, device=hidden.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D features [N, F*T], got shape {tuple(x.shape)}.")
        if x.shape[1] % self.d_feat != 0:
            raise ValueError(
                f"Feature dimension {x.shape[1]} is not divisible by d_feat={self.d_feat}. "
                "For Alpha360 use d_feat=6; for Alpha158 use d_feat=158."
            )
        # [N, F*T] -> [N, T, F]
        x = x.reshape(x.shape[0], self.d_feat, -1).permute(0, 2, 1)
        output, _ = self.encoder(x)
        return output[:, -1, :]

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.encode(x)

        if adj is None:
            if self.adj_mode == "dynamic":
                adj = self.build_dynamic_adj(hidden)
            elif self.adj_mode == "full":
                adj = self.build_full_adj(hidden)
            elif self.adj_mode == "identity":
                adj = self.build_identity_adj(hidden)
            else:
                raise ValueError("adj_mode='external' requires an adjacency matrix in forward(..., adj=...).")

        z = self.activation(self.gcn1(hidden, adj))
        z = self.dropout(z)
        z = self.activation(self.gcn2(z, adj))

        if self.add_residual:
            z = torch.cat([hidden, z], dim=-1)
        return self.head(z).squeeze(-1)


class QlibGCN(Model):
    """Qlib forecast model wrapper for TemporalGCNNet."""

    METRICS = {"", "loss", "mse", "ic"}

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        gcn_hidden_size: int = 64,
        dropout: float = 0.1,
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        metric: str = "ic",
        early_stop: int = 20,
        loss: str = "mse",
        base_model: str = "GRU",
        adj_mode: str = "dynamic",
        topk: int = 20,
        min_edge_weight: float = 0.0,
        relation_path: Optional[str] = None,
        optimizer: str = "adam",
        GPU: Union[int, str] = 0,
        seed: Optional[int] = None,
        clip_grad: Optional[float] = 3.0,
        valid_key=DataHandlerLP.DK_L,
        **kwargs,
    ):
        self.logger = get_module_logger("QlibGCN")
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gcn_hidden_size = gcn_hidden_size
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.metric = metric.lower() if metric else ""
        self.early_stop = early_stop
        self.loss = loss.lower()
        self.base_model = base_model
        self.adj_mode = adj_mode
        self.topk = topk
        self.min_edge_weight = min_edge_weight
        self.relation_path = relation_path
        self.optimizer = optimizer.lower()
        self.seed = seed
        self.clip_grad = clip_grad
        self.valid_key = valid_key
        self.extra_kwargs = kwargs

        self.device = self._resolve_device(GPU)

        if self.loss != "mse":
            raise ValueError("This implementation supports loss='mse' only.")
        if self.metric not in self.METRICS:
            raise ValueError(f"metric must be one of {sorted(self.METRICS)}.")
        if self.adj_mode == "external" and not relation_path:
            raise ValueError("relation_path must be provided when adj_mode='external'.")

        self._set_seed(seed)
        self.relation_graph = self._load_relation_graph(relation_path) if relation_path else None
        self.gcn_model = TemporalGCNNet(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gcn_hidden_size=gcn_hidden_size,
            dropout=dropout,
            base_model=base_model,
            adj_mode=adj_mode,
            topk=topk,
            min_edge_weight=min_edge_weight,
        ).to(self.device)

        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer in {"sgd", "gd"}:
            self.train_optimizer = optim.SGD(self.gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported.")

        self.fitted = False
        self.best_epoch = None
        self.logger.info(
            "QlibGCN parameters: "
            f"d_feat={d_feat}, hidden_size={hidden_size}, num_layers={num_layers}, "
            f"gcn_hidden_size={gcn_hidden_size}, dropout={dropout}, n_epochs={n_epochs}, "
            f"lr={lr}, metric={metric}, early_stop={early_stop}, base_model={base_model}, "
            f"adj_mode={adj_mode}, topk={topk}, device={self.device}, seed={seed}"
        )

    @property
    def use_gpu(self) -> bool:
        return self.device.type == "cuda"

    @staticmethod
    def _resolve_device(GPU: Union[int, str, None]) -> torch.device:
        if GPU is None:
            return torch.device("cpu")
        if isinstance(GPU, str):
            gpu = GPU.strip().lower()
            if gpu in {"", "none", "cpu", "-1"}:
                return torch.device("cpu")
            if gpu.isdigit():
                GPU = int(gpu)
            else:
                return torch.device(gpu)
        return torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and int(GPU) >= 0 else "cpu")

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _load_relation_graph(path: Optional[str]) -> Optional[pd.DataFrame]:
        if path is None:
            return None
        graph_path = Path(path).expanduser()
        if not graph_path.exists():
            raise FileNotFoundError(f"relation_path not found: {graph_path}")

        suffix = graph_path.suffix.lower()
        if suffix == ".csv":
            graph = pd.read_csv(graph_path, index_col=0)
        elif suffix in {".pkl", ".pickle"}:
            graph = pd.read_pickle(graph_path)
        elif suffix == ".parquet":
            graph = pd.read_parquet(graph_path)
        elif suffix == ".npy":
            arr = np.load(graph_path)
            graph = pd.DataFrame(arr)
        else:
            raise ValueError("relation_path must be csv, pkl, pickle, parquet, or npy.")

        graph.index = graph.index.astype(str)
        graph.columns = graph.columns.astype(str)
        return graph.astype("float32")

    @staticmethod
    def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex):
            return df.sort_index()
        raise ValueError("QlibGCN expects a MultiIndex dataframe indexed by [datetime, instrument].")

    @staticmethod
    def _to_numpy_feature(x: pd.DataFrame) -> np.ndarray:
        values = x.to_numpy(dtype="float32", copy=False)
        if not values.flags.writeable:
            values = values.copy()
        if values.ndim != 2:
            raise ValueError(f"Feature must be 2D, got shape {values.shape}.")
        return values

    @staticmethod
    def _to_numpy_label(y: pd.DataFrame) -> np.ndarray:
        values = y.to_numpy(dtype="float32", copy=False)
        if not values.flags.writeable:
            values = values.copy()
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError(f"Only single-label regression is supported, got label shape {values.shape}.")
            values = values[:, 0]
        return values

    @staticmethod
    def _daily_slices(df: pd.DataFrame, shuffle: bool = False) -> list[tuple[int, int]]:
        if df.empty:
            return []
        counts = df.groupby(level=0, sort=False).size().to_numpy()
        starts = np.r_[0, np.cumsum(counts[:-1])]
        slices = [(int(start), int(count)) for start, count in zip(starts, counts)]
        if shuffle:
            random.shuffle(slices)
        return slices

    @staticmethod
    def _instruments_from_index(index: pd.Index) -> list[str]:
        if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
            raise ValueError("Expected MultiIndex with instrument at level 1 or named 'instrument'.")
        if "instrument" in index.names:
            instruments = index.get_level_values("instrument")
        else:
            instruments = index.get_level_values(1)
        return [str(inst) for inst in instruments]

    def _external_adj(self, instruments: Iterable[str]) -> torch.Tensor:
        if self.relation_graph is None:
            raise ValueError("External graph was requested, but relation_graph is not loaded.")
        instruments = list(instruments)
        adj_np = self.relation_graph.reindex(index=instruments, columns=instruments).fillna(0.0).to_numpy(dtype="float32")
        adj_np = np.maximum(adj_np, adj_np.T)
        adj = torch.as_tensor(adj_np, dtype=torch.float32, device=self.device)
        adj = TemporalGCNNet.drop_self_edges(adj)
        return TemporalGCNNet.normalize_adj(adj)

    def _loss_fn(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(label) & torch.isfinite(pred)
        if mask.sum() == 0:
            # Avoid NaN loss if a daily batch contains no valid label.
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], label[mask])

    def _metric_fn(self, pred: torch.Tensor, label: torch.Tensor) -> float:
        mask = torch.isfinite(label) & torch.isfinite(pred)
        if mask.sum() < 2:
            return float("nan")
        pred = pred[mask]
        label = label[mask]

        if self.metric in {"", "loss", "mse"}:
            return -float(F.mse_loss(pred, label).detach().cpu().item())

        pred_centered = pred - pred.mean()
        label_centered = label - label.mean()
        denom = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(label_centered**2)).clamp_min(1e-12)
        return float((torch.sum(pred_centered * label_centered) / denom).detach().cpu().item())

    def _forward_batch(self, feature: torch.Tensor, instruments: Optional[list[str]] = None) -> torch.Tensor:
        if self.adj_mode == "external":
            if instruments is None:
                raise ValueError("instruments are required for external adjacency.")
            adj = self._external_adj(instruments)
            return self.gcn_model(feature, adj=adj)
        return self.gcn_model(feature)

    def _iter_daily_batches(
        self,
        x_data: pd.DataFrame,
        y_data: Optional[pd.DataFrame] = None,
        shuffle: bool = False,
    ) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[str]]]]:
        x_values = self._to_numpy_feature(x_data)
        y_values = self._to_numpy_label(y_data) if y_data is not None else None

        for start, count in self._daily_slices(x_data, shuffle=shuffle):
            batch = slice(start, start + count)
            feature = torch.from_numpy(x_values[batch]).to(self.device)
            label = torch.from_numpy(y_values[batch]).to(self.device) if y_values is not None else None
            instruments = self._instruments_from_index(x_data.index[batch]) if self.adj_mode == "external" else None
            yield feature, label, instruments

    def _train_epoch(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
        self.gcn_model.train()
        losses: list[float] = []

        for feature, label, instruments in self._iter_daily_batches(x_train, y_train, shuffle=True):
            pred = self._forward_batch(feature, instruments)
            loss = self._loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.gcn_model.parameters(), self.clip_grad)
            self.train_optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        return float(np.nanmean(losses)) if losses else float("nan")

    def _test_epoch(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> tuple[float, float]:
        self.gcn_model.eval()
        losses: list[float] = []
        scores: list[float] = []

        with torch.no_grad():
            for feature, label, instruments in self._iter_daily_batches(x_data, y_data, shuffle=False):
                pred = self._forward_batch(feature, instruments)
                losses.append(float(self._loss_fn(pred, label).detach().cpu().item()))
                scores.append(self._metric_fn(pred, label))

        return float(np.nanmean(losses)), float(np.nanmean(scores))

    def fit(self, dataset: DatasetH, evals_result: Optional[dict] = None, save_path: Optional[str] = None):
        if evals_result is None:
            evals_result = {}

        has_valid = hasattr(dataset, "segments") and "valid" in dataset.segments
        if has_valid:
            df_train, df_valid = dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=self.valid_key,
            )
            df_valid = self._sort_df(df_valid)
        else:
            df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=self.valid_key)
            df_valid = None

        df_train = self._sort_df(df_train)
        if df_train.empty or (has_valid and df_valid is not None and df_valid.empty):
            raise ValueError("Empty data from dataset. Check market, segments, and data handler config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        if has_valid and df_valid is not None:
            x_valid, y_valid = df_valid["feature"], df_valid["label"]
        else:
            x_valid, y_valid = x_train, y_train

        evals_result["train"] = []
        evals_result["valid"] = []
        stop_steps = 0
        best_score = -np.inf
        best_state = copy.deepcopy(self.gcn_model.state_dict())
        best_epoch = 0

        self.logger.info("training...")
        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._train_epoch(x_train, y_train)
            train_eval_loss, train_score = self._test_epoch(x_train, y_train)
            valid_loss, valid_score = self._test_epoch(x_valid, y_valid)

            evals_result["train"].append(train_score)
            evals_result["valid"].append(valid_score)
            self.logger.info(
                f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, "
                f"train_eval_loss={train_eval_loss:.6f}, train_score={train_score:.6f}, "
                f"valid_loss={valid_loss:.6f}, valid_score={valid_score:.6f}"
            )

            score_for_selection = valid_score
            if np.isnan(score_for_selection):
                score_for_selection = -valid_loss

            if score_for_selection > best_score:
                best_score = score_for_selection
                best_epoch = epoch
                stop_steps = 0
                best_state = copy.deepcopy(self.gcn_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.best_epoch = best_epoch
        self.gcn_model.load_state_dict(best_state)
        self.fitted = True
        self.logger.info(f"best score: {best_score:.6f} @ epoch {best_epoch}")

        if save_path is not None:
            save_path = get_or_create_path(save_path)
            torch.save(best_state, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test = self._sort_df(x_test)
        index = x_test.index
        preds: list[np.ndarray] = []

        self.gcn_model.eval()
        with torch.no_grad():
            for feature, _, instruments in self._iter_daily_batches(x_test):
                pred = self._forward_batch(feature, instruments).detach().cpu().numpy()
                preds.append(pred)

        pred_values = np.concatenate(preds, axis=0) if preds else np.array([], dtype="float32")
        return pd.Series(pred_values.reshape(-1), index=index, name="score")

    def save(self, filename: str, **kwargs) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.gcn_model.state_dict(),
                "fitted": self.fitted,
                "best_epoch": self.best_epoch,
            },
            filename,
        )

    def load(self, filename: str, **kwargs) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.gcn_model.load_state_dict(checkpoint["state_dict"])
            self.fitted = bool(checkpoint.get("fitted", True))
            self.best_epoch = checkpoint.get("best_epoch")
        else:
            self.gcn_model.load_state_dict(checkpoint)
            self.fitted = True
