import importlib
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path


REQUIRED_PACKAGES = ("numpy", "pandas", "torch")
MISSING_PACKAGES = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]


def install_qlib_stubs() -> None:
    if importlib.util.find_spec("qlib") is not None:
        return

    qlib = types.ModuleType("qlib")
    qlib_data = types.ModuleType("qlib.data")
    qlib_dataset = types.ModuleType("qlib.data.dataset")
    qlib_handler = types.ModuleType("qlib.data.dataset.handler")
    qlib_log = types.ModuleType("qlib.log")
    qlib_model = types.ModuleType("qlib.model")
    qlib_model_base = types.ModuleType("qlib.model.base")
    qlib_utils = types.ModuleType("qlib.utils")

    class DatasetH:
        pass

    class DataHandlerLP:
        DK_L = "learn"
        DK_I = "infer"

    class Model:
        pass

    def get_module_logger(name):
        logger = logging.getLogger(name)
        logger.addHandler(logging.NullHandler())
        return logger

    def get_or_create_path(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    qlib_dataset.DatasetH = DatasetH
    qlib_handler.DataHandlerLP = DataHandlerLP
    qlib_log.get_module_logger = get_module_logger
    qlib_model_base.Model = Model
    qlib_utils.get_or_create_path = get_or_create_path

    sys.modules.update(
        {
            "qlib": qlib,
            "qlib.data": qlib_data,
            "qlib.data.dataset": qlib_dataset,
            "qlib.data.dataset.handler": qlib_handler,
            "qlib.log": qlib_log,
            "qlib.model": qlib_model,
            "qlib.model.base": qlib_model_base,
            "qlib.utils": qlib_utils,
        }
    )


def load_model_module():
    if MISSING_PACKAGES:
        raise unittest.SkipTest(f"missing packages: {', '.join(MISSING_PACKAGES)}")
    install_qlib_stubs()
    return importlib.import_module("qlib_gcn_model.qlib_gcn_model")


class FakeDataset:
    def __init__(self, frames):
        self.frames = frames
        self.segments = {key: None for key in frames}

    def prepare(self, segments, col_set=None, data_key=None):
        if isinstance(segments, list):
            return [self.prepare(segment, col_set=col_set, data_key=data_key) for segment in segments]
        frame = self.frames[segments]
        if col_set == "feature":
            return frame["feature"]
        return frame


@unittest.skipIf(MISSING_PACKAGES, f"missing packages: {', '.join(MISSING_PACKAGES)}")
class QlibGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_model_module()
        cls.np = importlib.import_module("numpy")
        cls.pd = importlib.import_module("pandas")
        cls.torch = importlib.import_module("torch")

    def make_frame(self, start, days=3, instruments=3, d_feat=2, steps=3):
        np = self.np
        pd = self.pd
        dates = pd.date_range(start, periods=days, freq="D")
        inst = [f"STK{i:03d}" for i in range(instruments)]
        index = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])

        feature_dim = d_feat * steps
        values = np.linspace(0.0, 1.0, len(index) * feature_dim, dtype="float32").reshape(len(index), feature_dim)
        labels = (values[:, :1] - values[:, -1:]).astype("float32")

        feature = pd.DataFrame(values, index=index, columns=[f"f{i}" for i in range(feature_dim)])
        label = pd.DataFrame(labels, index=index, columns=["label"])
        return pd.concat({"feature": feature, "label": label}, axis=1)

    def test_forward_returns_one_score_per_node(self):
        model = self.mod.TemporalGCNNet(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            gcn_hidden_size=3,
            dropout=0.0,
            topk=1,
        )
        x = self.torch.randn(5, 6)

        pred = model(x)

        self.assertEqual(tuple(pred.shape), (5,))
        self.assertTrue(self.torch.isfinite(pred).all())

    def test_daily_slices_handles_empty_dataframe(self):
        pd = self.pd
        index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        df = pd.DataFrame(index=index)

        self.assertEqual(self.mod.QlibGCN._daily_slices(df), [])

    def test_fit_and_predict_with_fake_dataset(self):
        frames = {
            "train": self.make_frame("2020-01-01"),
            "valid": self.make_frame("2020-02-01"),
            "test": self.make_frame("2020-03-01", days=2),
        }
        dataset = FakeDataset(frames)
        model = self.mod.QlibGCN(
            d_feat=2,
            hidden_size=4,
            num_layers=1,
            gcn_hidden_size=3,
            dropout=0.0,
            n_epochs=2,
            early_stop=2,
            metric="loss",
            topk=1,
            GPU=-1,
            seed=7,
        )

        model.fit(dataset)
        pred = model.predict(dataset)

        self.assertTrue(model.fitted)
        self.assertEqual(len(pred), len(frames["test"]))
        self.assertEqual(pred.name, "score")


if __name__ == "__main__":
    unittest.main()
