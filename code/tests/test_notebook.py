"""Read-only notebook JSON/AST checks; never execute notebook cells or import ML packages."""
import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "code" / "lab_day2.ipynb"
RUN_FLAGS = {
    "RUN_SANITY_CHECKS", "RUN_BASELINE", "RUN_BACKBONES", "RUN_TRAINING_ABLATIONS",
    "RUN_INFERENCE_EXPERIMENTS", "RUN_LATENCY", "RUN_FINAL_TRAINING", "RUN_FINAL_TEST",
    "RUN_PREPARE_DATA", "RUN_EDA", "RUN_REPO_SETUP", "RUN_EVAL_REPORTS",
    "RUN_EXPORT_RESULTS", "RUN_ARTIFACT_CHECK", "INSTALL_DEPS", "USE_GOOGLE_DRIVE",
}


class TestNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [c for c in cls.nb["cells"] if c["cell_type"] == "code"]
        cls.sources = ["".join(c["source"]) for c in cls.code_cells]
        cls.trees = [ast.parse(s) for s in cls.sources]
        cls.text = "\n".join("".join(c["source"]) for c in cls.nb["cells"])
        cls.code = "\n".join(cls.sources)

    def assignments(self, name):
        return [n.value for t in self.trees for n in ast.walk(t)
                if isinstance(n, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in n.targets)]

    def test_json_format_and_clean_outputs(self):
        self.assertEqual(self.nb["nbformat"], 4)
        ids = [c["id"] for c in self.nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.code_cells:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_every_code_cell_compiles_without_execution(self):
        for cell, source in zip(self.code_cells, self.sources):
            # All cells use Python subprocess rather than shell/magic syntax.
            compile(source, f"{NOTEBOOK.name}:{cell['id']}", "exec")

    def test_required_sections(self):
        headings = ["".join(c["source"]).splitlines()[0]
                    for c in self.nb["cells"] if c["cell_type"] == "markdown"]
        for number in range(19):
            self.assertTrue(any(h.startswith(f"# {number}. ") for h in headings), number)
        for phrase in ("Colab/Kaggle", "MD5", "EDA", "Sanity check", "baseline T00",
                       "backbone", "huấn luyện", "suy luận", "độ trễ", "validation",
                       "nhiều seed", "Final test", "score/grade", "results.xlsx", "artifact"):
            self.assertIn(phrase.lower(), self.text.lower())

    def test_flags_are_unique_and_off(self):
        for flag in RUN_FLAGS | {"ALLOW_CPU_TRAINING", "CONFIRM_NO_TEST_PREDICTIONS"}:
            values = self.assignments(flag)
            self.assertEqual(len(values), 1, flag)
            self.assertIs(ast.literal_eval(values[0]), False, flag)
        self.assertEqual(ast.literal_eval(self.assignments("CONFIRM_FINAL_TEST")[0]), "NO")
        for field in ("SELECTED_BACKBONE", "FINAL_BACKBONE", "FINAL_TRAINING_CONFIG",
                      "FINAL_INFERENCE_METHOD", "COMBINED_RECIPE"):
            self.assertIsNone(ast.literal_eval(self.assignments(field)[0]))

    def test_expensive_top_level_calls_are_flag_guarded(self):
        dangerous = {"run_record", "setup_repo", "evaluate_validation_method", "fit_temperature",
                     "predict_logits", "latency_report", "tta_latency", "mount", "safe_extract",
                     "copy2", "copytree", "ExcelWriter", "write_json_new"}

        def visit(nodes, guarded=False):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, ast.If):
                    names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                    visit(node.body, guarded or bool(names & RUN_FLAGS))
                    visit(node.orelse, guarded)
                    continue
                if isinstance(node, ast.Call):
                    function = node.func
                    name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
                    is_command = isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "subprocess" and name == "run"
                    if name in dangerous or is_command:
                        self.assertTrue(guarded, f"Unguarded costly call: {name}")
                visit(list(ast.iter_child_nodes(node)), guarded)

        # globals().get gate in the early repo cell is also statically off.
        for tree in self.trees:
            for node in tree.body:
                if isinstance(node, ast.If) and 'globals().get' in ast.unparse(node.test):
                    self.assertIn("RUN_REPO_SETUP", ast.unparse(node.test))
                    self.assertIn("False", ast.unparse(node.test))
                else:
                    visit([node])

    def test_baseline_and_training_never_enable_test_predictions(self):
        settings = [kw.value for tree in self.trees for node in ast.walk(tree)
                    if isinstance(node, ast.Call) for kw in node.keywords
                    if kw.arg == "save_test_predictions"]
        self.assertTrue(settings)
        self.assertTrue(all(isinstance(v, ast.Constant) and v.value is False for v in settings))
        self.assertIn("if cfg.save_test_predictions:", self.code)
        self.assertNotRegex(self.code, r"save_test_predictions\s*=\s*True")
        variants = ast.literal_eval(self.assignments("RECIPE_VARIANTS")[0])
        self.assertTrue(all("save_test_predictions" not in patch for _, _, patch in variants))

    def test_five_backbones_and_shared_formula(self):
        backbones = ast.literal_eval(self.assignments("BACKBONES")[0])
        self.assertEqual(backbones, [("B01", "resnet50"), ("B02", "resnext50_32x4d"),
                                    ("B03", "convnext_tiny"), ("B04", "deit_small_patch16_224"),
                                    ("B05", "efficientnet_b0")])
        self.assertIn('config_for(exp_id, backbone=backbone)', self.code)

    def test_platform_paths_and_dependencies(self):
        for name in ("IN_COLAB", "IN_KAGGLE", "PLATFORM", "PLATFORM_MODE", "OUTPUT_DIR"):
            self.assertTrue(self.assignments(name), name)
        for item in ("/kaggle/input/<dien-dataset-slug>", "/kaggle/working", "/content/deepweeds_data",
                     "timm", "fvcore", "openpyxl", "pytest", '"--ff-only", "origin", "main"'):
            self.assertIn(item, self.code)
        self.assertNotRegex(self.code, r'"pip", "install"[^\n]*"(?:torch|torchvision)"')

    def test_dataset_integrity_and_fold_zero(self):
        required = ast.literal_eval(self.assignments("REQUIRED_LABELS")[0])
        self.assertEqual(required, ["labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"])
        self.assertEqual(ast.literal_eval(self.assignments("EXPECTED_IMAGES")[0]), 17509)
        self.assertEqual(ast.literal_eval(self.assignments("EXPECTED_MD5")[0]),
                         "b7b30f96d466fba86016aa5a26606e0f")
        self.assertIn("stream.read(8 * 1024 * 1024)", self.code)
        self.assertIn("expected_total=EXPECTED_IMAGES", self.code)
        self.assertIn("len(jpgs) != EXPECTED_IMAGES", self.code)
        self.assertNotRegex(self.code, r"subset[1-4]")

    def test_inference_is_validation_only_before_final_gate(self):
        for cell in self.code_cells:
            source = "".join(cell["source"])
            if cell["id"] == "inference-helper":
                self.assertIn('data.load_split(LABELS_DIR, fold=0)[1]', source)
                self.assertNotIn("test_loader", source)
                self.assertIn("fit_temperature(logits, labels)", source)
        specs = ast.literal_eval(self.assignments("INFERENCE_SPECS")[0])
        self.assertTrue({"I00", "I01", "I02", "I03", "I05", "I06", "I07", "I08"} <= specs.keys())

    def test_final_double_confirmation_and_one_test_pass(self):
        source = next("".join(c["source"]) for c in self.code_cells if c["id"] == "final-test-code")
        self.assertTrue(source.startswith("if RUN_FINAL_TEST:"))
        self.assertIn('CONFIRM_FINAL_TEST != "I_HAVE_LOCKED_THE_CONFIGURATION"', source)
        self.assertIn("not CONFIRM_NO_TEST_PREDICTIONS", source)
        self.assertIn("any(p.exists()", source)
        self.assertLess(source.index("write_json_new(marker"), source.index("inference.predict_logits"))
        self.assertEqual(source.count("inference.predict_logits("), 1)
        self.assertNotIn("fit_temperature", source)
        self.assertIn('validation["temperature"]', source)
        self.assertEqual(ast.literal_eval(self.assignments("FINAL_SEEDS")[0]), [0, 1, 2])

    def test_eval_and_seven_workbook_sheets(self):
        self.assertIn("eval.py score", self.text)
        self.assertIn("eval.py grade", self.text)
        sheets = ast.literal_eval(self.assignments("SHEET_COLUMNS")[0])
        self.assertEqual(set(sheets), {"Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"})
        for item in ("freeze_panes", "auto_filter", "number_format", "CONFIRM_OVERWRITE_RESULTS", "engine=\"openpyxl\""):
            self.assertIn(item, self.code)

    def test_no_canned_metrics_or_secrets(self):
        metric_keys = {"macro_f1_val", "top1_val", "ece_val", "p50", "p95", "p99", "macro_f1", "top1"}
        for tree in self.trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value in metric_keys:
                            self.assertFalse(isinstance(value, ast.Constant) and isinstance(value.value, (int, float)),
                                             f"Hardcoded metric: {key.value}")
        self.assertIn("Nhận xét sau khi chạy:", self.text)
        self.assertNotRegex(self.text, r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
        self.assertNotRegex(self.text, r"(?i)(password|api_key|access_token)\s*=\s*['\"][^'\"]+['\"]")
        self.assertNotRegex(self.text, r"[A-Z]:[/\\](?:Users|AIinActoin)|/home/[^/]+/")

    def test_completed_modules_have_no_implementation_stubs(self):
        for name in ("dataset", "model", "losses", "train", "inference", "benchmark"):
            source = (ROOT / "code" / f"{name}.py").read_text(encoding="utf-8")
            tree = ast.parse(source)
            self.assertFalse(any(isinstance(node, ast.Name) and node.id == "NotImplementedError"
                                 for node in ast.walk(tree)), name)
            self.assertNotIn("TODO", source, name)


if __name__ == "__main__":
    unittest.main()
