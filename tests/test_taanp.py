import sys
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

from args import initialize_args, get_experiment_id
from experiments import epoch_default, epoch_task_weighted
from model import get_model
from utils.evaluation_tool import mask_beta_elbo


def model_args(*extra):
    argv = ["test", "-model", "TAANP", "-dev", "cpu",
            "-h_dim", "16", "-z_dim", "16", "-num_heads", "4",
            "-npp", "both", "-opp", "normal", "-loss", "mask_beta_elbo", *extra]
    with patch.object(sys, "argv", argv):
        return initialize_args()


def example_batch():
    graphs = []
    for i in range(2):
        graphs.append(Data(
            num_nodes=3, y=torch.rand(3, 8, 1),
            static_aux=torch.rand(3, 8, 12),
            dynamic_aux=torch.rand(3, 8, 4), time_enc=torch.rand(3, 8, 6),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_attr=torch.ones(3),
            label_unobs_mask=torch.tensor([False, False, True]),
            n_id=torch.arange(3), t_id=torch.tensor([i]),
        ))
    return Batch.from_data_list(graphs)


class SyntheticLoader(list):
    scale = 1.0

    def inverse_transform(self, values):
        return values


class TAANPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_missing_context_is_excluded_from_both_training_losses(self):
        for epoch_module in [epoch_default, epoch_task_weighted]:
            with self.subTest(epoch=epoch_module.__name__):
                torch.manual_seed(42)
                args = model_args()
                model = get_model(args.model)(args)
                batch = example_batch()
                # First graph has no observed history; second has one missing value.
                batch.y[:2, :4] = torch.nan
                batch.y[3, 0] = torch.nan
                expected_mask = ~batch.y.isnan()
                expected_mask[:3] = False
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                before = [p.detach().clone() for p in model.parameters()]
                masks = []

                def loss_with_mask(preds, targets, mask, **kwargs):
                    masks.append(mask.detach().clone())
                    return mask_beta_elbo(preds, targets, mask, **kwargs)

                weighted_loss = epoch_task_weighted.task_weighted_beta_elbo

                def weighted_with_mask(**kwargs):
                    masks.append(kwargs["valid_mask"].detach().clone())
                    return weighted_loss(**kwargs)

                with patch.object(epoch_module, "log_performance",
                                  side_effect=lambda **kw: kw["metrics"]), \
                     patch.object(epoch_task_weighted, "task_weighted_beta_elbo",
                                  side_effect=weighted_with_mask):
                    epoch_module.shared_epoch(
                        model, SyntheticLoader([batch]), optimizer,
                        {args.loss: loss_with_mask}, args, "train", True, 0,
                    )
                self.assertEqual(len(masks), 1)
                torch.testing.assert_close(masks[0], expected_mask)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(any(not torch.equal(old, new)
                                    for old, new in zip(before, model.parameters())))

    def test_feature_ablations_keep_both_paths(self):
        variants = [
            [], ["--no-use-static"], ["--no-use-hit"], ["--no-use-speed"],
            ["--no-use-hit", "--no-use-speed"],
        ]
        for options in variants:
            with self.subTest(options=options):
                args = model_args(*options)
                model = get_model(args.model)(args).eval()
                with torch.no_grad():
                    output = model(example_batch())
                distribution, prior, posterior = output
                self.assertIsInstance(prior, torch.distributions.Normal)
                self.assertIsInstance(posterior, torch.distributions.Normal)
                self.assertEqual(distribution.loc.shape, (6, 8, 1))
                self.assertTrue(torch.isfinite(distribution.loc).all())
                self.assertTrue((distribution.scale > 0).all())

    def test_single_path_modes_are_rejected(self):
        for mode in ("latent", "deterministic"):
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    model_args("-npp", mode)
                self.assertEqual(error.exception.code, 2)

    def test_diy_keywords_do_not_change_model_behavior(self):
        models = []
        for note in ["experiment", "poi_not_static_not_hit_not_speed_gcn_dynamic_sparse"]:
            args = model_args("-diy", note)
            torch.manual_seed(42)
            models.append(get_model(args.model)(args).eval())
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        batch = example_batch()
        outputs = []
        for model in models:
            torch.manual_seed(123)
            with torch.no_grad():
                outputs.append(model(batch, num_sampling=30))
        for first, second in zip(*outputs):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_linear_model_does_not_require_graph_edges(self):
        batch = example_batch()
        without_edges = batch.clone()
        del without_edges.edge_index
        del without_edges.edge_attr
        for options in [[], ["--without_tamqm"]]:
            with self.subTest(options=options):
                model = get_model("TAANP")(model_args(*options)).eval()
                outputs = []
                for inputs in (batch, without_edges):
                    torch.manual_seed(123)
                    with torch.no_grad():
                        outputs.append(model(inputs, num_sampling=30))
                for first, second in zip(*outputs):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_explicit_feature_flags_and_experiment_names(self):
        base = model_args("-diy", "same-note")
        variant = model_args("-diy", "same-note", "--no-use-hit", "--use-poi")
        model = get_model(variant.model)(variant)
        self.assertTrue(model.not_hit)
        self.assertTrue(model.poi)
        self.assertFalse(model.not_speed)
        self.assertNotEqual(get_experiment_id(base), get_experiment_id(variant))
        weighted = model_args("-diy", "same-note", "--without_tamqm",
                              "-etype", "task_weighted", "-tw", "4", "1", "1")
        self.assertEqual(weighted.diy, "same-note")
        self.assertTrue(get_experiment_id(weighted).endswith("without_tamqm-tw-4-1-1"))

    def test_parser_defaults_match_the_published_model(self):
        with patch.object(sys, "argv", ["test"]):
            args = initialize_args()
        self.assertEqual((args.np_path, args.output_path), ("both", "normal"))
        self.assertEqual((args.hidden_dim, args.z_dim, args.num_heads), (64, 64, 4))
        self.assertTrue(args.use_static and args.use_hit and args.use_speed)
        self.assertFalse(args.use_poi or args.without_tamqm)

    def test_without_tamqm_keeps_shared_query_attention_and_sampling(self):
        args = model_args()
        args.without_tamqm = True
        model = get_model(args.model)(args)
        self.assertIn("cross_attention.q_proj.weight", model.state_dict())
        batch = example_batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        output = model(batch)
        loss = mask_beta_elbo(output, batch.y, ~batch.y.isnan(), beta=0.1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, alea, epist = model(batch, num_sampling=30)
        self.assertEqual(mean.shape, batch.y.shape)
        self.assertTrue(all(torch.isfinite(x).all() for x in (mean, alea, epist)))


if __name__ == "__main__":
    unittest.main()
