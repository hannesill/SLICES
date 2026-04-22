"""Tests for FairnessEvaluator.

Tests cover:
- Per-group AUROC computation with known predictions
- Worst-group AUROC selection
- Age binning: 30->"18-44", 50->"45-64", 70->"65-79", 85->"80+"
- Attribute auto-detection: gender+age available, race not
- Min subgroup size filtering: groups with <50 samples excluded
- Evaluate with synthetic data where one group has perfect predictions
- Evaluate returns correct structure
"""

import polars as pl
import pytest
import torch
from slices.eval.fairness_evaluator import FairnessEvaluator, flatten_fairness_report


def make_static_df(
    n: int = 100,
    include_race: bool = False,
    genders: list = None,
    ages: list = None,
) -> pl.DataFrame:
    """Create a synthetic static DataFrame for testing."""
    if genders is None:
        genders = ["M" if i % 2 == 0 else "F" for i in range(n)]
    if ages is None:
        ages = [30 + (i % 60) for i in range(n)]

    data = {
        "stay_id": list(range(n)),
        "patient_id": list(range(n)),
        "gender": genders,
        "age": ages,
        "los_days": [5.0] * n,
    }
    if include_race:
        data["race"] = [
            "White" if i % 3 == 0 else "Black" if i % 3 == 1 else "Asian" for i in range(n)
        ]

    return pl.DataFrame(data)


class TestPerGroupAUROC:
    """Tests for per-group AUROC computation."""

    def test_known_predictions(self):
        """Per-group AUROC should reflect group-level performance."""
        n = 200
        static_df = make_static_df(n=n)
        stay_ids = list(range(n))

        # Group 0 (M, even indices): perfect predictions
        # Group 1 (F, odd indices): random predictions
        labels = torch.zeros(n)
        predictions = torch.zeros(n)

        for i in range(n):
            labels[i] = 1.0 if i % 4 < 2 else 0.0
            if i % 2 == 0:  # Male
                predictions[i] = 0.9 if labels[i] == 1 else 0.1
            else:  # Female
                predictions[i] = 0.5  # Random-ish

        evaluator = FairnessEvaluator(
            static_df, protected_attributes=["gender"], min_subgroup_size=10
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        assert "gender" in report
        per_group = report["gender"]["per_group_auroc"]
        # Male group should have higher AUROC
        assert per_group.get("M", 0) > per_group.get("F", 1)

    def test_perfect_predictions_both_groups(self):
        """Both groups with perfect predictions -> AUROC ~1.0 for both."""
        n = 200
        static_df = make_static_df(n=n)
        stay_ids = list(range(n))

        # Labels: 25% cycle (1,1,0,0) so both M (even) and F (odd) get both classes
        labels = torch.tensor([1.0 if i % 4 < 2 else 0.0 for i in range(n)])
        predictions = torch.tensor([0.95 if labels[i] == 1 else 0.05 for i in range(n)])

        evaluator = FairnessEvaluator(
            static_df, protected_attributes=["gender"], min_subgroup_size=10
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        for group, auroc in report["gender"]["per_group_auroc"].items():
            assert auroc > 0.9, f"Group {group} AUROC={auroc}, expected > 0.9"


class TestWorstGroupAUROC:
    """Tests for worst-group AUROC selection."""

    def test_selects_minimum(self):
        """Worst-group should be the minimum per-group AUROC."""
        n = 200
        static_df = make_static_df(n=n)
        stay_ids = list(range(n))

        labels = torch.zeros(n)
        predictions = torch.zeros(n)

        for i in range(n):
            labels[i] = 1.0 if i % 4 < 2 else 0.0
            if i % 2 == 0:  # Male - good predictions
                predictions[i] = 0.9 if labels[i] == 1 else 0.1
            else:  # Female - worse predictions
                predictions[i] = 0.6 if labels[i] == 1 else 0.4

        evaluator = FairnessEvaluator(
            static_df, protected_attributes=["gender"], min_subgroup_size=10
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        worst = report["gender"]["worst_group_auroc"]
        per_group = report["gender"]["per_group_auroc"]
        valid_aurocs = [v for v in per_group.values() if v == v]  # Filter NaN
        assert worst == pytest.approx(min(valid_aurocs))


class TestRegressionFairness:
    """Tests for regression-task fairness metrics."""

    def test_singleton_prediction_dimension_is_squeezed(self):
        """Regression fairness should treat (N, 1) predictions as samplewise outputs."""
        static_df = make_static_df(
            n=4,
            genders=["M", "M", "F", "F"],
            ages=[50.0, 50.0, 50.0, 50.0],
        )
        stay_ids = [0, 1, 2, 3]
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        predictions = torch.tensor([[0.0], [1.5], [0.0], [0.5]])

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender"],
            min_subgroup_size=1,
            task_type="regression",
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        per_group_mse = report["gender"]["per_group_mse"]
        per_group_mae = report["gender"]["per_group_mae"]
        assert per_group_mse["M"] == pytest.approx(0.125)
        assert per_group_mse["F"] == pytest.approx(0.125)
        assert per_group_mae["M"] == pytest.approx(0.25)
        assert per_group_mae["F"] == pytest.approx(0.25)
        assert report["gender"]["worst_group_mse"] == pytest.approx(0.125)
        assert report["gender"]["worst_group_mae"] == pytest.approx(0.25)
        assert report["gender"]["mse_gap"] == pytest.approx(0.0)
        assert report["gender"]["mae_gap"] == pytest.approx(0.0)


class TestAgeBinning:
    """Tests for age binning."""

    def test_age_bins(self):
        """Check specific ages are binned correctly."""
        evaluator = FairnessEvaluator.__new__(FairnessEvaluator)
        ages = torch.tensor([30.0, 50.0, 70.0, 85.0])
        groups = evaluator._bin_age(ages)

        # 30 -> 18-44 (index 0)
        assert groups[0].item() == 0
        # 50 -> 45-64 (index 1)
        assert groups[1].item() == 1
        # 70 -> 65-79 (index 2)
        assert groups[2].item() == 2
        # 85 -> 80+ (index 3)
        assert groups[3].item() == 3

    def test_boundary_ages(self):
        """Test bin boundaries."""
        evaluator = FairnessEvaluator.__new__(FairnessEvaluator)
        ages = torch.tensor([18.0, 44.0, 45.0, 64.0, 65.0, 79.0, 80.0])
        groups = evaluator._bin_age(ages)

        assert groups[0].item() == 0  # 18 -> 18-44
        assert groups[1].item() == 0  # 44 -> 18-44
        assert groups[2].item() == 1  # 45 -> 45-64
        assert groups[3].item() == 1  # 64 -> 45-64
        assert groups[4].item() == 2  # 65 -> 65-79
        assert groups[5].item() == 2  # 79 -> 65-79
        assert groups[6].item() == 3  # 80 -> 80+

    def test_null_and_nan_ages_are_unknown(self):
        """Null and numeric NaN ages should not fall into the youngest bin."""
        static_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [1, 2, 3],
                "gender": ["M", "F", "M"],
                "age": [None, float("nan"), 30.0],
                "los_days": [5.0, 5.0, 5.0],
            }
        )
        evaluator = FairnessEvaluator(static_df, protected_attributes=["age_group"])

        group_ids, group_names, _ = evaluator._encode_attribute([1, 2, 3], "age_group")

        assert group_ids.tolist() == [-1, -1, 0]
        assert group_names[-1] == "unknown"


class TestAttributeAutoDetection:
    """Tests for attribute auto-detection."""

    def test_gender_and_age_available(self):
        """When gender and age columns exist, both should be detected."""
        static_df = make_static_df(include_race=False)
        evaluator = FairnessEvaluator(static_df)
        assert "gender" in evaluator._available_attributes
        assert "age_group" in evaluator._available_attributes
        assert "race" not in evaluator._available_attributes

    def test_race_available_when_present(self):
        """When race column exists, it should be detected."""
        static_df = make_static_df(include_race=True)
        evaluator = FairnessEvaluator(
            static_df, protected_attributes=["gender", "age_group", "race"]
        )
        assert "race" in evaluator._available_attributes

    def test_missing_columns_filtered(self):
        """Requested attributes without corresponding columns are skipped."""
        static_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [1, 2, 3],
                "los_days": [5.0, 5.0, 5.0],
            }
        )
        evaluator = FairnessEvaluator(static_df, protected_attributes=["gender", "race"])
        assert evaluator._available_attributes == []


class TestMinSubgroupSize:
    """Tests for min_subgroup_size filtering."""

    def test_small_groups_excluded(self):
        """Groups below min_subgroup_size should be excluded."""
        # 90 male, 10 female
        genders = ["M"] * 90 + ["F"] * 10
        ages = [50.0] * 100
        static_df = make_static_df(n=100, genders=genders, ages=ages)
        stay_ids = list(range(100))

        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(100)])
        predictions = torch.tensor([0.8 if i % 2 == 0 else 0.2 for i in range(100)])

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender"],
            min_subgroup_size=50,
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        # Female group (n=10) should be excluded -> no gender report
        assert "gender" not in report

    def test_all_groups_large_enough(self):
        """All groups above min_subgroup_size should be included."""
        genders = ["M"] * 100 + ["F"] * 100
        ages = [50.0] * 200
        static_df = make_static_df(n=200, genders=genders, ages=ages)
        stay_ids = list(range(200))

        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(200)])
        predictions = torch.tensor([0.8 if i % 2 == 0 else 0.2 for i in range(200)])

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender"],
            min_subgroup_size=50,
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        assert "gender" in report
        assert len(report["gender"]["per_group_auroc"]) == 2

    def test_threshold_counts_unique_patients_not_stays(self):
        """Min subgroup size should be enforced on patients, not repeated stays."""
        rows = []
        stay_id = 0
        for patient_id in range(40):
            for _ in range(2):
                rows.append(
                    {
                        "stay_id": stay_id,
                        "patient_id": f"m-{patient_id}",
                        "gender": "M",
                        "age": 50.0,
                        "los_days": 5.0,
                    }
                )
                stay_id += 1
        for patient_id in range(55):
            rows.append(
                {
                    "stay_id": stay_id,
                    "patient_id": f"f-{patient_id}",
                    "gender": "F",
                    "age": 50.0,
                    "los_days": 5.0,
                }
            )
            stay_id += 1

        static_df = pl.DataFrame(rows)
        predictions = torch.tensor([0.8 if i % 2 == 0 else 0.2 for i in range(len(rows))])
        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(len(rows))])
        stay_ids = static_df["stay_id"].to_list()

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender"],
            min_subgroup_size=50,
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        # Male has 80 stays but only 40 patients, so the attribute should be skipped.
        assert "gender" not in report


class TestEvaluateStructure:
    """Tests for evaluate() return structure."""

    def test_returns_correct_keys(self):
        """Evaluate should return expected nested structure."""
        n = 200
        static_df = make_static_df(n=n)
        stay_ids = list(range(n))

        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
        predictions = torch.tensor([0.7 if i % 2 == 0 else 0.3 for i in range(n)])

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender"],
            min_subgroup_size=10,
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        assert "gender" in report
        gender_report = report["gender"]
        assert "per_group_auroc" in gender_report
        assert "worst_group_auroc" in gender_report
        assert "demographic_parity_diff" in gender_report
        assert "equalized_odds_diff" in gender_report
        assert "disparate_impact_ratio" in gender_report
        assert "n_valid_groups" in gender_report
        assert "group_sizes" in gender_report

    def test_multiple_attributes(self):
        """Should compute metrics for multiple attributes."""
        n = 400
        # Need diverse ages across bins with enough per bin
        ages = []
        for i in range(n):
            if i % 4 == 0:
                ages.append(30.0)
            elif i % 4 == 1:
                ages.append(50.0)
            elif i % 4 == 2:
                ages.append(70.0)
            else:
                ages.append(85.0)

        static_df = make_static_df(n=n, ages=ages)
        stay_ids = list(range(n))

        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
        predictions = torch.tensor([0.7 if i % 2 == 0 else 0.3 for i in range(n)])

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender", "age_group"],
            min_subgroup_size=10,
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        assert "gender" in report
        assert "age_group" in report

    def test_group_sizes_report_patients_and_samples_separately(self):
        """The report should expose patient counts separately from stay counts."""
        rows = []
        stay_id = 0
        for patient_id in range(55):
            for _ in range(2):
                rows.append(
                    {
                        "stay_id": stay_id,
                        "patient_id": f"m-{patient_id}",
                        "gender": "M",
                        "age": 50.0,
                        "los_days": 5.0,
                    }
                )
                stay_id += 1
        for patient_id in range(55):
            for _ in range(2):
                rows.append(
                    {
                        "stay_id": stay_id,
                        "patient_id": f"f-{patient_id}",
                        "gender": "F",
                        "age": 50.0,
                        "los_days": 5.0,
                    }
                )
                stay_id += 1

        static_df = pl.DataFrame(rows)
        predictions = torch.tensor([0.8 if i % 2 == 0 else 0.2 for i in range(len(rows))])
        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(len(rows))])
        stay_ids = static_df["stay_id"].to_list()

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["gender"],
            min_subgroup_size=50,
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        assert report["gender"]["group_sizes"]["M"] == 55
        assert report["gender"]["group_sample_sizes"]["M"] == 110

    def test_race_uses_canonical_miiv_only_schema_for_combined(self):
        """Combined race fairness should only include canonical MIMIC race bins."""
        static_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "patient_id": ["a", "b", "c", "d"],
                "gender": ["M", "F", "M", "F"],
                "age": [50.0, 60.0, 55.0, 65.0],
                "race": [
                    "WHITE",
                    "BLACK/AFRICAN AMERICAN",
                    "HISPANIC/LATINO",
                    "ASIAN",
                ],
                "source_dataset": ["miiv", "miiv", "eicu", "eicu"],
                "los_days": [5.0, 5.0, 5.0, 5.0],
            }
        )
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        stay_ids = [1, 2, 3, 4]

        evaluator = FairnessEvaluator(
            static_df,
            protected_attributes=["race"],
            min_subgroup_size=1,
            dataset_name="combined",
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)

        assert "race" in report
        assert set(report["race"]["per_group_auroc"]) == {"White", "Black"}


class TestPrintReport:
    """Tests for print_report()."""

    def test_print_without_error(self, capsys):
        """print_report should run without errors."""
        n = 200
        static_df = make_static_df(n=n)
        stay_ids = list(range(n))

        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
        predictions = torch.tensor([0.7 if i % 2 == 0 else 0.3 for i in range(n)])

        evaluator = FairnessEvaluator(
            static_df, protected_attributes=["gender"], min_subgroup_size=10
        )
        report = evaluator.evaluate(predictions, labels, stay_ids)
        evaluator.print_report(report)

        captured = capsys.readouterr()
        assert "Fairness Evaluation Report" in captured.out
        assert "gender" in captured.out


class TestFlattenFairnessReport:
    """Tests for fairness report flattening."""

    def test_flattens_nested_metrics_for_logging(self):
        report = {
            "gender": {
                "worst_group_auroc": 0.71,
                "per_group_auroc": {"M": 0.83, "F": 0.71},
                "group_sizes": {"M": 55, "F": 60},
            }
        }

        flat = flatten_fairness_report(report)

        assert flat["fairness/gender/worst_group_auroc"] == pytest.approx(0.71)
        assert flat["fairness/gender/per_group_auroc/M"] == pytest.approx(0.83)
        assert flat["fairness/gender/per_group_auroc/F"] == pytest.approx(0.71)
        assert flat["fairness/gender/group_sizes/M"] == 55
        assert flat["fairness/gender/group_sizes/F"] == 60
