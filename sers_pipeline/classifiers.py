"""Module 3 — Classifiers.

Two tiers, by design:

    Tier A — PCA + SVC (or RandomForest), with SHAP-based explainability.
             This is what you actually present to clinical reviewers. PCA
             gives noise-robust feature compression; SVC is small, fast, and
             well-understood. SHAP values back-projected through PCA loadings
             give a per-wavenumber importance map that says "the model called
             this positive because of intensity at 1245 and 1580 cm^-1", which
             a reviewer can sanity-check against known molecular vibrations.

    Tier B — 1D CNN (RamanNet-style).
             Deeper models can squeeze out a few % extra accuracy on hard
             classes but are harder to explain. Use Tier B for sensitivity
             gains, Tier A for the explainability story. In practice you'll
             run both in parallel and gate the final call on Tier A unless
             Tier B disagrees strongly.

Pretrained weights:
    Papers like RamanNet (Pan et al.) and DeepeR (Ho et al., 2019) publish
    architectures; pretrained weights are sometimes released on GitHub but
    availability is inconsistent. The safer assumption is that you train
    from scratch on your own labeled data + synthetic pretraining. We expose
    a `load_pretrained()` hook for when/if you find compatible weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# Tier A: PCA + SVC, with SHAP explainability                                  #
# --------------------------------------------------------------------------- #

@dataclass
class TierAClassifier:
    """PCA-compressed SVC with probability calibration.

    `probability=True` enables Platt scaling so we get calibrated
    `predict_proba` outputs — required for the grey-zone safety gate.
    """
    n_components: int = 20
    C: float = 1.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    random_state: int = 0
    _pipeline: Optional[SkPipeline] = field(default=None, init=False, repr=False)
    _classes: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TierAClassifier":
        self._pipeline = SkPipeline([
            ("pca", PCA(n_components=self.n_components, random_state=self.random_state)),
            ("svc", SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=True,
                random_state=self.random_state,
            )),
        ])
        self._pipeline.fit(X, y)
        self._classes = np.array(self._pipeline.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._classes

    def save(self, path: str | Path) -> None:
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "TierAClassifier":
        inst = cls()
        inst._pipeline = joblib.load(path)
        inst._classes = np.array(inst._pipeline.classes_)
        return inst

    # ----- Explainability -------------------------------------------------- #
    def wavenumber_importance(
        self,
        X_background: np.ndarray,
        X_query: np.ndarray,
        wavenumbers: np.ndarray,
        n_background_samples: int = 50,
    ) -> np.ndarray:
        """Per-wavenumber SHAP-style contribution map for each query sample.

        Approach: explain the SVC on PCA features, then back-project SHAP
        values through PCA loadings to recover a per-wavenumber importance.

        Returns
        -------
        (n_query, n_classes, n_wavenumbers) array of contributions.

        Notes
        -----
        For a clinical audit, the auditable claim is: "the prediction was
        driven by intensity at these specific wavenumbers." Back-projection
        through PCA preserves this linear interpretation because PCA is
        itself linear; the only nonlinearity sits in the SVC kernel and is
        captured by SHAP's local explanation.
        """
        import shap  # local import — heavy dep

        pca = self._pipeline.named_steps["pca"]
        svc = self._pipeline.named_steps["svc"]

        # Background and query in PCA space
        bg = pca.transform(X_background[:n_background_samples])
        Q = pca.transform(X_query)

        explainer = shap.KernelExplainer(svc.predict_proba, bg)
        shap_vals = explainer.shap_values(Q, silent=True)
        # shap_vals: list of (n_query, n_components) arrays, one per class
        # OR (n_query, n_components, n_classes) depending on SHAP version
        shap_arr = np.asarray(shap_vals)
        if shap_arr.ndim == 3 and shap_arr.shape[0] == len(self._classes):
            # (n_classes, n_query, n_components) → transpose
            shap_arr = shap_arr.transpose(1, 0, 2)
        elif shap_arr.ndim == 3:
            # (n_query, n_components, n_classes) → transpose
            shap_arr = shap_arr.transpose(0, 2, 1)
        # Now: (n_query, n_classes, n_components)

        # Back-project: (n_components, n_wavenumbers) loadings
        loadings = pca.components_
        # contribution = shap[component] * loading[component, wavenumber]
        contrib = shap_arr @ loadings   # (n_query, n_classes, n_wavenumbers)
        return contrib


# --------------------------------------------------------------------------- #
# Tier B: 1D CNN (RamanNet-style)                                              #
# --------------------------------------------------------------------------- #

# torch is an optional-heavy import. Keep it inside the class so the rest of
# the module is usable without torch installed.

class TierBClassifier:
    """1D CNN for spectral classification.

    Architecture rationale:
        * Narrow conv kernels (size 5-21) match Raman peak widths in pixels
        * Multiple conv blocks of increasing channel depth — early layers
          learn peak shapes, later layers learn peak combinations
        * Global average pooling instead of dense flatten — much fewer
          parameters and translation-equivariant w.r.t. small wavenumber
          shifts (helps with calibration drift)
        * Dropout in classifier head for regularization

    Not a clone of any specific paper — this is a small, defensible default.
    Swap layer counts / channel widths as your dataset grows.
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        random_state: int = 0,
        verbose: bool = False,
    ):
        import torch
        torch.manual_seed(random_state)
        self.torch = torch
        self.n_classes = n_classes
        self.n_features = n_features
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.model = self._build_model()
        self._fitted = False

    def _build_model(self):
        torch = self.torch
        nn = torch.nn

        class RamanCNN(nn.Module):
            """1D CNN preserving positional information.

            Earlier draft used AdaptiveAvgPool1d(1) — a global average that
            destroys position. For Raman, peak *position* is the discriminative
            signal, so we keep a coarse position axis via AdaptiveAvgPool1d(16)
            and flatten that into the classifier head.
            """
            def __init__(self, n_classes: int):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=9, padding=4),
                    nn.BatchNorm1d(16), nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),

                    nn.Conv1d(16, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),

                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                    # Preserve 16 positional bins (not 1!) so the classifier
                    # head can use *where* peaks are, not just their average.
                    nn.AdaptiveAvgPool1d(16),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),                          # 64 * 16 = 1024
                    nn.Dropout(0.3),
                    nn.Linear(64 * 16, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, n_classes),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return RamanCNN(self.n_classes).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TierBClassifier":
        torch = self.torch
        X_t = torch.from_numpy(X).float().unsqueeze(1).to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            running, n_correct, n_total = 0.0, 0, 0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(yb)
                n_correct += (logits.argmax(1) == yb).sum().item()
                n_total += len(yb)
            scheduler.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1:3d}  loss={running/n_total:.4f}  "
                      f"train_acc={n_correct/n_total:.3f}")
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().unsqueeze(1).to(self.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        self.torch.save({
            "state_dict": self.model.state_dict(),
            "n_classes": self.n_classes,
            "n_features": self.n_features,
        }, path)

    def load_pretrained(self, path: str | Path, strict: bool = True) -> None:
        """Load externally-published weights (DeepeR, RamanNet, custom).

        `strict=False` lets you partially load when only the convolutional
        feature extractor matches and the classification head differs — a
        common transfer-learning pattern when adapting to a new analyte set.
        """
        ckpt = self.torch.load(path, map_location=self.device)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(sd, strict=strict)
        self._fitted = True

    # ----- Explainability -------------------------------------------------- #
    def wavenumber_importance(
        self,
        X_background: np.ndarray,
        X_query: np.ndarray,
        n_background_samples: int = 32,
    ) -> np.ndarray:
        """Per-wavenumber importance for the CNN via SHAP gradient explainer.

        Uses `shap.GradientExplainer` because it works directly on the conv
        net's gradients — much faster than KernelExplainer on a deep model.
        Background samples come from the training distribution; query
        samples are the spectra you actually want to explain.

        Returns
        -------
        (n_query, n_classes, n_wavenumbers) array of attributions.
        Positive values: this wavenumber pushed the prediction toward that
        class; negative: pushed away.

        Notes
        -----
        Pair this with the SHAP attributions from Tier A. When both classifiers
        agree on which wavenumbers drove a prediction, your audit story is
        much stronger. When they disagree, that's worth investigating — it
        often signals overfitting in one of them.
        """
        import shap
        torch = self.torch

        self.model.eval()
        bg = torch.from_numpy(X_background[:n_background_samples]).float().unsqueeze(1).to(self.device)
        Q = torch.from_numpy(X_query).float().unsqueeze(1).to(self.device)

        explainer = shap.GradientExplainer(self.model, bg)
        # shap_values: list of (n_query, 1, n_features) one per class
        shap_vals = explainer.shap_values(Q)
        arr = np.asarray(shap_vals)
        # Normalize the array shape to (n_query, n_classes, n_features)
        if arr.ndim == 4 and arr.shape[0] == self.n_classes:
            # (n_classes, n_query, 1, n_features)
            arr = arr.squeeze(2).transpose(1, 0, 2)
        elif arr.ndim == 4:
            # (n_query, 1, n_features, n_classes)
            arr = arr.squeeze(1).transpose(0, 2, 1)
        return arr
