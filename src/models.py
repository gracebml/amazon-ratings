"""
Pure NumPy Matrix Factorization for Collaborative Filtering.

Implements SVD-based recommendation system without using external ML libraries.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional

import numpy as np


# DATA LOADING

def load_raw_csv_ratings(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ratings from CSV file. Returns (user_ids, item_ids, ratings)."""
    user_ids: list[str] = []
    item_ids: list[str] = []
    ratings: list[float] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_ids.append(row["UserId"])
            item_ids.append(row["ProductId"])
            ratings.append(float(row["Rating"]))

    return (
        np.array(user_ids, dtype=str),
        np.array(item_ids, dtype=str),
        np.array(ratings, dtype=np.float32),
    )


def load_numpy_ratings(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ratings from CSV path or directory containing ratings_Beauty.csv."""
    if path.lower().endswith(".csv"):
        csv_path = path
    else:
        csv_path = os.path.join(path, "ratings_Beauty.csv")
    return load_raw_csv_ratings(csv_path)


def filter_top_users(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter to keep only ratings from top-k most active users."""
    if top_k is None or top_k <= 0:
        return user_ids, item_ids, ratings, np.unique(user_ids)

    unique_users, counts = np.unique(user_ids, return_counts=True)
    if top_k >= unique_users.size:
        return user_ids, item_ids, ratings, unique_users

    order = np.argsort(counts)[::-1]
    top_users = unique_users[order[:top_k]]
    mask = np.isin(user_ids, top_users)
    return user_ids[mask], item_ids[mask], ratings[mask], top_users


def _get_field(row: np.void, *candidates: str):
    """Case-insensitive field lookup in structured numpy array."""
    dtype_names = row.dtype.names or ()
    lookup = {name.lower(): name for name in dtype_names}
    for name in candidates:
        actual = lookup.get(name.lower())
        if actual is not None:
            return row[actual]
    raise KeyError(f"None of {candidates} found in dtype names {dtype_names}")


def _zscore_features(matrix: np.ndarray) -> np.ndarray:
    """Standardize features to zero mean and unit variance."""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-8
    return (matrix - mean) / std


def load_user_feature_dict(npz_path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load user features from NPZ file.
    Returns (feature_dict, feature_dimension).
    """
    data = np.load(npz_path)["data"]
    ids: list[str] = []
    feats: list[list[float]] = []
    for row in data:
        uid = str(_get_field(row, "UserId", "user_id", "userID"))
        ids.append(uid)
        feats.append(
            [
                float(
                    _get_field(
                        row,
                        "num_ratings",
                        "user_num_ratings",
                        "ratings_count",
                    )
                ),
                float(_get_field(row, "avg_rating", "user_avg_rating")),
                float(
                    _get_field(
                        row,
                        "positive_ratio",
                        "user_positive_ratio",
                    )
                ),
            ]
        )

    feat_matrix = _zscore_features(np.asarray(feats, dtype=np.float32))
    feature_dict = {uid: feat_matrix[i] for i, uid in enumerate(ids)}
    return feature_dict, feat_matrix.shape[1]


def load_item_feature_dict(npz_path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load item features from NPZ file.
    Returns (feature_dict, feature_dimension).
    """
    data = np.load(npz_path)["data"]
    ids: list[str] = []
    feats: list[list[float]] = []
    for row in data:
        iid = str(_get_field(row, "ProductId", "product_id", "asin"))
        ids.append(iid)
        feats.append(
            [
                float(
                    _get_field(
                        row,
                        "num_ratings",
                        "product_num_ratings",
                        "ratings_count",
                    )
                ),
                float(_get_field(row, "avg_rating", "product_avg_rating")),
                float(
                    _get_field(
                        row,
                        "popularity_score",
                        "product_popularity",
                    )
                ),
            ]
        )

    feat_matrix = _zscore_features(np.asarray(feats, dtype=np.float32))
    feature_dict = {iid: feat_matrix[i] for i, iid in enumerate(ids)}
    return feature_dict, feat_matrix.shape[1]


# TRAIN/TEST SPLIT

def group_shuffle_split(
    user_ids: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data by user groups to avoid data leakage.
    Each user's ratings go entirely to train or test set.
    """
    unique_users = np.unique(user_ids)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_users)

    test_user_count = max(1, int(len(unique_users) * test_ratio))
    test_users = set(unique_users[:test_user_count])
    mask = np.isin(user_ids, list(test_users))

    test_idx = np.where(mask)[0]
    train_idx = np.where(~mask)[0]
    return train_idx, test_idx


# EVALUATION METRICS

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate RMSE and MAE for rating predictions."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return {"RMSE": rmse, "MAE": mae}


# MATRIX FACTORIZATION MODEL

@dataclass
class PureNumpyMFConfig:
    """Configuration for Pure NumPy Matrix Factorization."""
    n_factors: int = 100
    n_epochs: int = 40
    lr: float = 0.005
    reg: float = 0.04
    random_state: int = 42
    use_user_features: bool = False
    use_item_features: bool = False


class PureNumpyMF:
    """
    Matrix Factorization with biases using pure NumPy and SGD.
    
    Prediction formula:
        r_hat = mu + b_u + b_i + p_u^T q_i + [optional: feature terms]
    
    Where:
        - mu: global mean rating
        - b_u, b_i: user and item biases
        - p_u, q_i: user and item latent factor vectors (size=n_factors)
    
    All operations are vectorized for performance. No Python loops for 
    matrix computations.
    """

    def __init__(self, config: PureNumpyMFConfig | None = None) -> None:
        self.config = config or PureNumpyMFConfig()

        # Model parameters (initialized in fit)
        self.global_mean: float = 0.0
        self.user_to_index: Dict[str, int] = {}
        self.item_to_index: Dict[str, int] = {}
        self.index_to_user: np.ndarray | None = None
        self.index_to_item: np.ndarray | None = None

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_feat_matrix: np.ndarray | None = None
        self.item_feat_matrix: np.ndarray | None = None
        self.user_feat_weights: np.ndarray | None = None
        self.item_feat_weights: np.ndarray | None = None

    def _encode_ids(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map string IDs to integer indices for efficient array indexing."""
        unique_users, user_idx = np.unique(user_ids, return_inverse=True)
        unique_items, item_idx = np.unique(item_ids, return_inverse=True)

        self.index_to_user = unique_users
        self.index_to_item = unique_items
        self.user_to_index = {uid: i for i, uid in enumerate(unique_users)}
        self.item_to_index = {iid: i for i, iid in enumerate(unique_items)}
        return user_idx, item_idx

    def _init_parameters(
        self,
        n_users: int,
        n_items: int,
        user_feat_dim: int,
        item_feat_dim: int,
    ) -> None:
        """Initialize model parameters using Glorot scaling for stability."""
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        dtype = np.float64
        
        # Glorot initialization: scale = sqrt(1 / n_factors)
        scale = np.sqrt(1.0 / cfg.n_factors)
        self.user_factors = rng.normal(0.0, scale, size=(n_users, cfg.n_factors)).astype(dtype)
        self.item_factors = rng.normal(0.0, scale, size=(n_items, cfg.n_factors)).astype(dtype)
        self.user_bias = np.zeros(n_users, dtype=dtype)
        self.item_bias = np.zeros(n_items, dtype=dtype)
        self.user_feat_weights = (
            np.zeros(user_feat_dim, dtype=dtype) if user_feat_dim > 0 else None
        )
        self.item_feat_weights = (
            np.zeros(item_feat_dim, dtype=dtype) if item_feat_dim > 0 else None
        )

    def _build_feature_matrix(
        self,
        index_to_id: np.ndarray,
        feature_dict: Dict[str, np.ndarray],
        n_entities: int,
        feat_dim: int,
    ) -> np.ndarray:
        """Vectorized construction of feature matrix from feature dict."""
        feat_matrix = np.zeros((n_entities, feat_dim), dtype=np.float64)
        
        # Vectorized lookup: build list of features in index order
        features_list = [
            feature_dict.get(entity_id, np.zeros(feat_dim, dtype=np.float64))
            for entity_id in index_to_id
        ]
        feat_matrix[:] = np.array(features_list, dtype=np.float64)
        return feat_matrix

    def _ensure_fitted(self) -> None:
        """Check if model has been fitted before prediction."""
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.user_bias is None
            or self.item_bias is None
        ):
            raise RuntimeError("Call fit() before predict/recommend.")

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        user_feature_dict: Optional[Dict[str, np.ndarray]] = None,
        item_feature_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Train the model using Stochastic Gradient Descent.
        
        All updates are vectorized using NumPy broadcasting and np.add.at
        for efficient in-place aggregation.
        """
        cfg = self.config
        user_idx, item_idx = self._encode_ids(user_ids, item_ids)
        ratings = ratings.astype(np.float64, copy=False)
        self.global_mean = float(ratings.mean())

        n_users = len(self.index_to_user) if self.index_to_user is not None else 0
        n_items = len(self.index_to_item) if self.index_to_item is not None else 0

        # Build feature matrices (vectorized, no loops)
        use_user_feats = bool(
            cfg.use_user_features and user_feature_dict and self.index_to_user is not None
        )
        user_feat_dim = 0
        if use_user_feats:
            sample_vec = next(iter(user_feature_dict.values()))
            user_feat_dim = len(sample_vec)
            self.user_feat_matrix = self._build_feature_matrix(
                self.index_to_user, user_feature_dict, n_users, user_feat_dim
            )
        else:
            self.user_feat_matrix = None

        use_item_feats = bool(
            cfg.use_item_features and item_feature_dict and self.index_to_item is not None
        )
        item_feat_dim = 0
        if use_item_feats:
            sample_vec = next(iter(item_feature_dict.values()))
            item_feat_dim = len(sample_vec)
            self.item_feat_matrix = self._build_feature_matrix(
                self.index_to_item, item_feature_dict, n_items, item_feat_dim
            )
        else:
            self.item_feat_matrix = None

        self._init_parameters(n_users, n_items, user_feat_dim, item_feat_dim)

        # SGD training loop
        rng = np.random.default_rng(cfg.random_state)
        n_samples = len(ratings)
        indices = np.arange(n_samples)
        
        for epoch in range(cfg.n_epochs):
            # Shuffle data for SGD
            rng.shuffle(indices)
            shuffled_users = user_idx[indices]
            shuffled_items = item_idx[indices]
            shuffled_ratings = ratings[indices]

            # Compute predictions (vectorized)
            preds = self._predict_from_indices(shuffled_users, shuffled_items, clip=False)
            err = shuffled_ratings - preds
            
            # Clip errors to prevent numerical overflow
            err = np.clip(err, -10.0, 10.0)

            # Update biases (vectorized with gradient clipping)
            user_bias_update = err - cfg.reg * self.user_bias[shuffled_users]
            item_bias_update = err - cfg.reg * self.item_bias[shuffled_items]
            np.add.at(self.user_bias, shuffled_users, cfg.lr * np.clip(user_bias_update, -10.0, 10.0))
            np.add.at(self.item_bias, shuffled_items, cfg.lr * np.clip(item_bias_update, -10.0, 10.0))

            # Update latent factors (vectorized with gradient clipping)
            user_factor_sel = self.user_factors[shuffled_users]
            item_factor_sel = self.item_factors[shuffled_items]
            
            user_factor_grad = err[:, None] * item_factor_sel - cfg.reg * user_factor_sel
            item_factor_grad = err[:, None] * user_factor_sel - cfg.reg * item_factor_sel
            
            user_factor_grad = np.clip(user_factor_grad, -5.0, 5.0)
            item_factor_grad = np.clip(item_factor_grad, -5.0, 5.0)
            
            np.add.at(self.user_factors, shuffled_users, cfg.lr * user_factor_grad)
            np.add.at(self.item_factors, shuffled_items, cfg.lr * item_factor_grad)

            # Update feature weights (vectorized matrix operations)
            if (
                self.user_feat_matrix is not None
                and self.user_feat_weights is not None
            ):
                user_feat_batch = self.user_feat_matrix[shuffled_users]
                grad = user_feat_batch.T @ err - cfg.reg * self.user_feat_weights
                self.user_feat_weights += cfg.lr * np.clip(grad, -5.0, 5.0)
            if (
                self.item_feat_matrix is not None
                and self.item_feat_weights is not None
            ):
                item_feat_batch = self.item_feat_matrix[shuffled_items]
                grad = item_feat_batch.T @ err - cfg.reg * self.item_feat_weights
                self.item_feat_weights += cfg.lr * np.clip(grad, -5.0, 5.0)

        # Return training metrics
        preds = self.predict(user_ids, item_ids)
        return regression_metrics(ratings, preds)

    def _predict_from_indices(
        self, user_idx: np.ndarray, item_idx: np.ndarray, clip: bool = False
    ) -> np.ndarray:
        """
        Vectorized prediction from integer indices.
        Formula: mu + b_u + b_i + p_u^T q_i + feature_terms
        """
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None

        # Base: global mean + biases (vectorized)
        base = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
        )
        
        # Interaction: dot product of latent factors (vectorized)
        interaction = np.sum(
            self.user_factors[user_idx] * self.item_factors[item_idx], axis=1
        )
        
        # Feature terms (vectorized matrix-vector products)
        feat_term = np.zeros_like(interaction, dtype=np.float64)
        if (
            self.user_feat_matrix is not None
            and self.user_feat_weights is not None
        ):
            feat_term += self.user_feat_matrix[user_idx] @ self.user_feat_weights
        if (
            self.item_feat_matrix is not None
            and self.item_feat_weights is not None
        ):
            feat_term += self.item_feat_matrix[item_idx] @ self.item_feat_weights
        
        result = base + interaction + feat_term
        if clip:
            return np.clip(result, 1.0, 5.0)
        return result

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Predict ratings for given (user, item) pairs.
        Returns global mean for unknown users/items (cold start).
        """
        self._ensure_fitted()
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float32)
        
        # Vectorized ID lookup
        user_idx = np.fromiter(
            (self.user_to_index.get(uid, -1) for uid in user_ids),
            dtype=np.int64,
            count=len(user_ids),
        )
        item_idx = np.fromiter(
            (self.item_to_index.get(iid, -1) for iid in item_ids),
            dtype=np.int64,
            count=len(item_ids),
        )
        
        # Predict only for known (user, item) pairs
        mask = (user_idx >= 0) & (item_idx >= 0)
        if np.any(mask):
            preds[mask] = self._predict_from_indices(
                user_idx[mask], item_idx[mask], clip=True
            )
        return preds

    def predict_one(self, user_id: str, item_id: str) -> float:
        """Predict single rating. Returns global mean for unknown user/item."""
        self._ensure_fitted()
        u_idx = self.user_to_index.get(user_id)
        i_idx = self.item_to_index.get(item_id)
        if u_idx is None or i_idx is None:
            return self.global_mean
        
        val = self._predict_from_indices(
            np.array([u_idx], dtype=np.int64),
            np.array([i_idx], dtype=np.int64),
            clip=True
        )[0]
        return float(np.clip(val, 1.0, 5.0))

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_items: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Recommend top-n items for a user.
        Returns structured array: [(item_id, predicted_rating), ...]
        """
        self._ensure_fitted()
        if user_id not in self.user_to_index:
            raise ValueError(f"Unknown user_id: {user_id}")

        u_idx = self.user_to_index[user_id]
        exclude = set() if exclude_items is None else set(exclude_items)
        
        assert self.index_to_item is not None
        
        # Vectorized prediction for all items
        item_indices = np.arange(len(self.index_to_item), dtype=np.int64)
        user_vector_idx = np.full_like(item_indices, u_idx, dtype=np.int64)
        scores = self._predict_from_indices(user_vector_idx, item_indices, clip=True)

        # Filter and sort (minimal Python loop, only for filtering)
        if exclude:
            mask = ~np.isin(self.index_to_item, list(exclude))
            filtered_items = self.index_to_item[mask]
            filtered_scores = scores[mask]
        else:
            filtered_items = self.index_to_item
            filtered_scores = scores
        
        # Sort by score (descending) and take top-n
        top_indices = np.argsort(filtered_scores)[::-1][:n]
        top_items = filtered_items[top_indices]
        top_scores = np.clip(filtered_scores[top_indices], 1.0, 5.0)
        
        # Return as structured array
        dtype = np.dtype([("product_id", "U32"), ("predicted_rating", "f4")])
        result = np.empty(len(top_items), dtype=dtype)
        result["product_id"] = top_items
        result["predicted_rating"] = top_scores
        return result


# TRAINING PIPELINE

def train_numpy_mf_pipeline(
    csv_path: str,
    test_ratio: float = 0.2,
    config: PureNumpyMFConfig | None = None,
    user_feature_npz: Optional[str] = None,
    item_feature_npz: Optional[str] = None,
    top_k_users: Optional[int] = None,
) -> Dict[str, float]:
    """
    Training pipeline for NumPy MF model.
    
    Args:
        csv_path: Path to ratings CSV file
        test_ratio: Fraction of users for test set
        config: Model hyperparameters
        user_feature_npz: Optional user features
        item_feature_npz: Optional item features
        top_k_users: Optional filter to top-k active users
    
    Returns:
        Test set metrics (RMSE, MAE)
    """
    # Load and optionally filter data
    users, items, ratings = load_raw_csv_ratings(csv_path)
    if top_k_users:
        users, items, ratings, _ = filter_top_users(users, items, ratings, top_k_users)
    
    # Split by user groups
    train_idx, test_idx = group_shuffle_split(users, test_ratio=test_ratio)

    train_users, train_items, train_ratings = (
        users[train_idx],
        items[train_idx],
        ratings[train_idx],
    )
    test_users, test_items, test_ratings = (
        users[test_idx],
        items[test_idx],
        ratings[test_idx],
    )

    # Load optional features
    user_feat_dict = None
    item_feat_dict = None
    if user_feature_npz:
        user_feat_dict, _ = load_user_feature_dict(user_feature_npz)
    if item_feature_npz:
        item_feat_dict, _ = load_item_feature_dict(item_feature_npz)

    # Train and evaluate
    model = PureNumpyMF(config)
    model.fit(
        train_users,
        train_items,
        train_ratings,
        user_feature_dict=user_feat_dict,
        item_feature_dict=item_feat_dict,
    )
    preds = model.predict(test_users, test_items)
    return regression_metrics(test_ratings, preds)
