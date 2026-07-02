"""Tests for SurvivalPredictor — Cox PH / Weibull AFT / RSF model selection.

Trains on tiny survival_df (30 rows) — lifelines and sksurv train in <2s.
"""
import numpy as np
import pandas as pd
import pytest

from pdm.survival.model import SurvivalPredictor
from pdm.base import TrainResult, PredictionResult


class TestSurvivalPredictor:
    def test_train_returns_concordance(self, survival_df):
        model = SurvivalPredictor()
        result = model.train(survival_df, survival_df)
        assert isinstance(result, TrainResult)
        assert "concordance_index" in result.metrics
        assert 0 <= result.metrics["concordance_index"] <= 1

    def test_model_type_selected(self, survival_df):
        model = SurvivalPredictor()
        model.train(survival_df, survival_df)
        assert model.model_type in ("cox", "weibull", "rsf")

    def test_predict_returns_result(self, survival_df):
        model = SurvivalPredictor()
        model.train(survival_df, survival_df)
        result = model.predict(survival_df)
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) > 0
        assert "median_survival" in result.predictions.columns

    def test_save_load_roundtrip(self, survival_df, tmp_model_dir):
        model = SurvivalPredictor()
        model.train(survival_df, survival_df)
        model.save(tmp_model_dir)

        loaded = SurvivalPredictor.load(tmp_model_dir)
        preds = loaded.predict(survival_df)
        assert len(preds.predictions) > 0

    def test_feature_names_exclude_target(self, survival_df):
        model = SurvivalPredictor()
        model.train(survival_df, survival_df)
        assert "duration" not in model.feature_names
        assert "event" not in model.feature_names
