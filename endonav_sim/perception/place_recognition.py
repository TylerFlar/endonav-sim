"""Online visual place recognition.

Tries DINOv2 ViT-S/14 + VLAD aggregation for descriptive features; falls back
to roll-invariant HSV histograms if torch / DINOv2 are unavailable.

Image is *de-rotated* by the camera's cumulative roll before feature
extraction so that the same anatomical viewpoint produces the same descriptor
regardless of how much the shaft has been twisted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Optional DINOv2 backend ----------------------------------------------------
_TORCH_OK = False
try:  # pragma: no cover - exercised opportunistically
    import torch
    import torch.nn.functional as F  # noqa: F401  (kept for parity if extended)

    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False


@dataclass
class MatchResult:
    best_match_node: str | None
    best_similarity: float
    all_similarities: dict[str, float]
    is_revisit: bool


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _kmeans_numpy(
    X: np.ndarray, k: int, n_iter: int = 25, seed: int = 0
) -> np.ndarray:
    """Lloyd's algorithm; X is (N, D); returns (k, D) centroids."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= k:
        # Pad by repeating points
        idx = rng.integers(0, n, size=k)
        return X[idx].copy()
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = X[init_idx].copy()
    for _ in range(n_iter):
        # Assign
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        new_centroids = centroids.copy()
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                new_centroids[ci] = X[mask].mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


class PlaceRecognition:
    def __init__(
        self,
        similarity_threshold: float = 0.80,
        novelty_threshold: float = 0.90,
        max_keyframes: int = 200,
        vlad_clusters: int = 32,
        vocab_warmup_frames: int = 50,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.novelty_threshold = float(novelty_threshold)
        self.max_keyframes = int(max_keyframes)
        self.K = int(vlad_clusters)
        self.vocab_warmup_frames = int(vocab_warmup_frames)

        self._nodes: dict[str, np.ndarray] = {}
        self._pending_features: dict[str, np.ndarray] = {}  # raw patch features
        self._vocab_features: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None

        self._backend = "hsv"
        self._dino = None
        self._dino_layer_out: list[np.ndarray] = []
        self._device = "cpu"

        if _TORCH_OK:
            try:
                model = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
                )
                model.eval()
                self._dino = model
                self._backend = "dinov2"

                # Hook the value projection of layer 9's attention.
                def _hook(_module, _inp, output):
                    # output of qkv linear: (B, N, 3*D)
                    out = output.detach().cpu().numpy()
                    B, N, three_d = out.shape
                    D = three_d // 3
                    v = out[:, :, 2 * D : 3 * D]  # (B, N, D)
                    self._dino_layer_out.append(v)

                model.blocks[9].attn.qkv.register_forward_hook(_hook)
            except Exception:
                self._dino = None
                self._backend = "hsv"

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    def _derotate(self, frame: np.ndarray, cumulative_roll: float) -> np.ndarray:
        H, W = frame.shape[:2]
        M = cv2.getRotationMatrix2D(
            (W / 2.0, H / 2.0), -float(np.rad2deg(cumulative_roll)), 1.0
        )
        rotated = cv2.warpAffine(frame, M, (W, H), borderValue=0)
        # Center-crop to a square inscribed in a circle of radius S/2 so that
        # *every* rotation angle samples the same disc of pixels. Without
        # this, content drifts in/out of the corners as the camera rolls and
        # CNN/transformer features become roll-sensitive.
        S = int(min(H, W) / np.sqrt(2))
        y0 = (H - S) // 2
        x0 = (W - S) // 2
        return rotated[y0 : y0 + S, x0 : x0 + S]

    # ------------------------------------------------------------------
    def _hsv_descriptor(self, frame_rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        d = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        return _l2_normalize(d)

    # ------------------------------------------------------------------
    def _dino_patch_features(self, frame_rgb: np.ndarray) -> np.ndarray:
        assert self._dino is not None
        img = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(
            np.float32
        ) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        self._dino_layer_out.clear()
        with torch.no_grad():
            self._dino(tensor)
        v = self._dino_layer_out[-1][0]  # (N, D)
        # Drop the CLS token (index 0)
        if v.shape[0] > 1:
            v = v[1:]
        return v.astype(np.float32)

    # ------------------------------------------------------------------
    def _vlad_encode(self, features: np.ndarray) -> np.ndarray:
        assert self._centroids is not None
        K, D = self._centroids.shape
        d2 = ((features[:, None, :] - self._centroids[None, :, :]) ** 2).sum(-1)
        assignments = np.argmin(d2, axis=1)
        vlad = np.zeros((K, D), dtype=np.float32)
        for ci in range(K):
            mask = assignments == ci
            if mask.any():
                residuals = features[mask] - self._centroids[ci]
                vlad[ci] = residuals.sum(axis=0)
        # Intra-normalization
        for ci in range(K):
            n = float(np.linalg.norm(vlad[ci]))
            if n > 1e-12:
                vlad[ci] /= n
        return _l2_normalize(vlad.flatten())

    # ------------------------------------------------------------------
    def _maybe_fit_vocab(self) -> None:
        if self._centroids is not None:
            return
        if len(self._vocab_features) < self.vocab_warmup_frames:
            return
        all_feat = np.concatenate(self._vocab_features, axis=0)
        if all_feat.shape[0] > 20000:
            idx = np.random.default_rng(0).choice(all_feat.shape[0], size=20000, replace=False)
            all_feat = all_feat[idx]
        self._centroids = _kmeans_numpy(all_feat, self.K, n_iter=25)
        self._vocab_features.clear()

    # ------------------------------------------------------------------
    def extract_descriptor(
        self, frame: np.ndarray, cumulative_roll: float = 0.0
    ) -> np.ndarray:
        canonical = self._derotate(frame, cumulative_roll)
        if self._backend == "dinov2":
            feats = self._dino_patch_features(canonical)
            self._vocab_features.append(feats)
            self._maybe_fit_vocab()
            if self._centroids is None:
                # Mean-pool placeholder so something is returned during warmup.
                d = feats.mean(axis=0)
                return _l2_normalize(d.astype(np.float32))
            return self._vlad_encode(feats)
        return self._hsv_descriptor(canonical)

    # ------------------------------------------------------------------
    def add_node(
        self, node_id: str, frame: np.ndarray, cumulative_roll: float = 0.0
    ) -> None:
        canonical = self._derotate(frame, cumulative_roll)
        if self._backend == "dinov2":
            feats = self._dino_patch_features(canonical)
            self._vocab_features.append(feats)
            self._pending_features[node_id] = feats
            self._maybe_fit_vocab()
            if self._centroids is None:
                self._nodes[node_id] = _l2_normalize(feats.mean(axis=0).astype(np.float32))
            else:
                self._nodes[node_id] = self._vlad_encode(feats)
        else:
            self._nodes[node_id] = self._hsv_descriptor(canonical)

        if len(self._nodes) > self.max_keyframes:
            # Drop oldest
            first = next(iter(self._nodes))
            self._nodes.pop(first, None)
            self._pending_features.pop(first, None)

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Recompute VLAD descriptors for any nodes added before vocab fit."""
        if self._backend != "dinov2":
            return
        self._maybe_fit_vocab()
        if self._centroids is None and self._pending_features:
            # Force vocab fit even if warmup count not reached
            all_feat = np.concatenate(list(self._pending_features.values()), axis=0)
            self._centroids = _kmeans_numpy(all_feat, self.K, n_iter=25)
        if self._centroids is None:
            return
        for node_id, feats in self._pending_features.items():
            self._nodes[node_id] = self._vlad_encode(feats)

    # ------------------------------------------------------------------
    def match(self, frame: np.ndarray, cumulative_roll: float = 0.0) -> MatchResult:
        desc = self.extract_descriptor(frame, cumulative_roll)
        sims: dict[str, float] = {}
        best_node: str | None = None
        best_sim = -1.0
        for node_id, ref in self._nodes.items():
            if ref.shape != desc.shape:
                # Mixed warmup/post-warmup descriptors — skip silently.
                continue
            s = float(np.dot(desc, ref))
            sims[node_id] = s
            if s > best_sim:
                best_sim = s
                best_node = node_id
        return MatchResult(
            best_match_node=best_node,
            best_similarity=float(best_sim) if best_node is not None else 0.0,
            all_similarities=sims,
            is_revisit=best_sim >= self.similarity_threshold,
        )
