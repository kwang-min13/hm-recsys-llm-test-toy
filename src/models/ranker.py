"""
LightGBM Ranker Module (Improved)

- Objective: lambdarank (listwise)
- Metric: ndcg@k (ranking quality)
- Requires group/query boundaries per user/session
- Includes strict validation checks (negatives must exist)
"""

from __future__ import annotations

import lightgbm as lgb
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PurchaseRanker:
    """구매/클릭 가능성 기반 Ranking 모델 (LambdaRank)"""

    def __init__(self):
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.params: Optional[Dict[str, Any]] = None

    @staticmethod
    def _to_numpy(X: pl.DataFrame) -> np.ndarray:
        return X.to_numpy()

    @staticmethod
    def _to_1d(y: pl.DataFrame) -> np.ndarray:
        arr = y.to_numpy()
        return arr.ravel()

    @staticmethod
    def _check_groups(group: List[int], n_rows: int, name: str) -> None:
        if not group or sum(group) != n_rows:
            raise ValueError(f"{name} group 합({sum(group) if group else 0}) != n_rows({n_rows}).")
        if any(g <= 0 for g in group):
            raise ValueError(f"{name} group에 0 이하 값이 있습니다: {group[:10]}")

    @staticmethod
    def _check_labels(y: np.ndarray, name: str) -> None:
        uniq = np.unique(y)
        if len(uniq) < 2:
            raise ValueError(
                f"{name}에 class가 1개뿐입니다(uniq={uniq}). "
                f"검증/학습에 negative 샘플이 반드시 포함되어야 합니다."
            )

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        group_train: List[int],
        X_val: pl.DataFrame,
        y_val: pl.DataFrame,
        group_val: List[int],
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
        eval_at: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Args:
            X_train, y_train: 학습 데이터 (후보 단위)
            group_train: 유저/세션별 후보 개수 리스트 (합 == len(X_train))
            X_val, y_val: 검증 데이터
            group_val: 검증 group
            params: LightGBM 파라미터
            eval_at: NDCG@K의 K들 (default [5,10,20])
        """
        logger.info("LambdaRank 학습 시작...")

        if eval_at is None:
            eval_at = [5, 10, 20]

        self.feature_names = list(X_train.columns)

        X_tr = self._to_numpy(X_train)
        y_tr = self._to_1d(y_train)
        X_va = self._to_numpy(X_val)
        y_va = self._to_1d(y_val)

        self._check_groups(group_train, len(X_tr), "train")
        self._check_groups(group_val, len(X_va), "valid")
        self._check_labels(y_tr, "train y")
        self._check_labels(y_va, "valid y")

        # 기본 파라미터: ranking용
        if params is None:
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "eval_at": eval_at,
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 127,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l1": 0.0,
                "lambda_l2": 1.0,
                "max_depth": -1,
                "verbosity": -1,
                "seed": 42,
            }

        self.params = params

        train_data = lgb.Dataset(
            X_tr,
            label=y_tr,
            group=group_train,
            feature_name=self.feature_names,
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_va,
            label=y_va,
            group=group_val,
            reference=train_data,
            feature_name=self.feature_names,
            free_raw_data=False,
        )

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=int(num_boost_round),
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )

        # best_score에서 valid ndcg@k 가져오기
        metrics: Dict[str, float] = {"best_iteration": float(self.model.best_iteration)}

        # LightGBM은 ndcg@k를 "ndcg@5" 같은 key로 저장
        for k in eval_at:
            tr_key = f"ndcg@{k}"
            if "train" in self.model.best_score and tr_key in self.model.best_score["train"]:
                metrics[f"train_{tr_key}"] = float(self.model.best_score["train"][tr_key])
            if "valid" in self.model.best_score and tr_key in self.model.best_score["valid"]:
                metrics[f"valid_{tr_key}"] = float(self.model.best_score["valid"][tr_key])

        logger.info("학습 완료. best_iteration=%s", self.model.best_iteration)
        return metrics

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict(self._to_numpy(X), num_iteration=self.model.best_iteration)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None or self.feature_names is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        imp = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, imp))

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_path = path.replace(".pkl", ".txt")
        self.model.save_model(model_path)

        metadata = {
            "feature_names": self.feature_names,
            "model_path": model_path,
            "params": self.params,
        }
        joblib.dump(metadata, path)
        logger.info("모델 저장 완료: %s", path)

    @classmethod
    def load(cls, path: str) -> "PurchaseRanker":
        ranker = cls()
        metadata = joblib.load(path)
        ranker.feature_names = metadata.get("feature_names")
        ranker.params = metadata.get("params")
        ranker.model = lgb.Booster(model_file=metadata["model_path"])
        logger.info("모델 로드 완료: %s", path)
        return ranker
