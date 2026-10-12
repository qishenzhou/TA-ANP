import math
import unittest

import torch

from experiments.epoch_task_weighted import (
    build_task_masks,
    task_weighted_beta_elbo,
)


class TaskWeightedEpochTest(unittest.TestCase):

    def test_task_masks_match_the_three_evaluation_subtasks(self):
        valid_mask = torch.ones(2, 8, 1, dtype=torch.bool)
        label_unobs_mask = torch.tensor([False, True])

        masks = build_task_masks(
            valid_mask=valid_mask,
            label_unobs_mask=label_unobs_mask,
            input_window_size=4,
            look_back_window_size=2,
            pred_window_size=4,
        )

        self.assertEqual(masks["OH"].sum().item(), 2)
        self.assertEqual(masks["OF"].sum().item(), 4)
        self.assertEqual(masks["UE"].sum().item(), 2)
        self.assertEqual(masks["UF"].sum().item(), 4)
        self.assertTrue(masks["OH"][0, 2:4].all())
        self.assertTrue(masks["OF"][0, 4:8].all())
        self.assertTrue(masks["UE"][1, 2:4].all())
        self.assertTrue(masks["UF"][1, 4:8].all())
        self.assertFalse(masks["OF"][0, :4].any())
        self.assertFalse(masks["UE"][1, :2].any())

    def test_task_nlls_are_normalized_before_weighting(self):
        targets = torch.zeros(2, 8, 1)
        loc = torch.zeros_like(targets)
        loc[0, :4] = 4.0    # OH, fixed weight 1
        loc[0, 4:8] = 1.0   # OF
        loc[1, :4] = 2.0    # UE
        loc[1, 4:8] = 3.0   # UF
        dist = torch.distributions.Normal(loc, torch.ones_like(loc))

        loss = task_weighted_beta_elbo(
            preds=dist,
            targets=targets,
            valid_mask=torch.ones_like(targets, dtype=torch.bool),
            label_unobs_mask=torch.tensor([False, True]),
            task_weights=[2.0, 1.0, 1.0],
            beta=1.0,
            input_window_size=4,
            look_back_window_size=4,
            pred_window_size=4,
        )

        gaussian_constant = 0.5 * math.log(2.0 * math.pi)
        expected = gaussian_constant + (8.0 + 2.0 * 0.5 + 2.0 + 4.5) / 5.0
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_kl_is_added_once_after_task_weighting(self):
        targets = torch.zeros(2, 2, 1)
        dist = torch.distributions.Normal(targets, torch.ones_like(targets))
        prior = torch.distributions.Normal(torch.zeros(1, 1), torch.ones(1, 1))
        posterior = torch.distributions.Normal(torch.ones(1, 1), torch.ones(1, 1))

        loss = task_weighted_beta_elbo(
            preds=(dist, prior, posterior),
            targets=targets,
            valid_mask=torch.ones_like(targets, dtype=torch.bool),
            label_unobs_mask=torch.tensor([False, True]),
            task_weights=[1.0, 1.0, 1.0],
            beta=0.25,
            input_window_size=1,
            look_back_window_size=1,
            pred_window_size=1,
        )

        gaussian_constant = 0.5 * math.log(2.0 * math.pi)
        expected = gaussian_constant + 0.25 * 0.5
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_empty_task_is_skipped_and_active_weights_are_renormalized(self):
        targets = torch.zeros(2, 2, 1)
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        valid_mask[1, 0] = False  # UE is empty; OH, OF, and UF remain valid.
        dist = torch.distributions.Normal(targets, torch.ones_like(targets))

        loss = task_weighted_beta_elbo(
            preds=dist,
            targets=targets,
            valid_mask=valid_mask,
            label_unobs_mask=torch.tensor([False, True]),
            task_weights=[1.0, 2.0, 1.0],
            beta=0.0,
            input_window_size=1,
            look_back_window_size=1,
            pred_window_size=1,
        )

        gaussian_constant = 0.5 * math.log(2.0 * math.pi)
        self.assertAlmostEqual(loss.item(), gaussian_constant, places=6)


if __name__ == "__main__":
    unittest.main()
