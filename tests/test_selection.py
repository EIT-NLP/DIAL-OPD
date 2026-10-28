"""CPU correctness checks for the score family; no model or GPU required."""
from __future__ import annotations

import math
import unittest

import torch

from dial_opd.selection import _log_scores, select_score_tokens

BETAS = (0., .25, .5, .75, 1., 1.5)


class ScoreSelectionTests(unittest.TestCase):
    def test_endpoints_match_reward_and_apd(self):
        p = torch.tensor([[.3, .04, .001, .2]], dtype=torch.float64)
        q = torch.tensor([[.2, .09, .005, .6]], dtype=torch.float64)
        student, teacher = p.log(), q.log()
        valid = torch.ones_like(p, dtype=torch.bool)
        for beta, reference in ((0., (teacher-student).abs()), (1., (q-p).abs())):
            selected, _ = select_score_tokens(student, teacher, valid, .5, beta)
            expected = torch.zeros_like(valid)
            expected[0, reference[0].argsort(descending=True, stable=True)[:2]] = True
            self.assertTrue(torch.equal(selected, expected))
        torch.testing.assert_close(_log_scores(student, teacher, 1.).exp(), (q-p).abs(),
                                   rtol=1e-14, atol=1e-16)

    def test_all_six_betas_follow_geometric_identity(self):
        student = torch.tensor([-1., -2., -7., -4.], dtype=torch.float64)
        teacher = torch.tensor([-2., -1.5, -8., -2.], dtype=torch.float64)
        r = (teacher-student).abs()
        d = (teacher.exp()-student.exp()).abs()
        for beta in BETAS:
            with self.subTest(beta=beta):
                got = _log_scores(student, teacher, beta)
                expected = (1-beta)*r.log()+beta*d.log()
                torch.testing.assert_close(got, expected, rtol=1e-14, atol=1e-14)
                torch.testing.assert_close(got.exp(), r.pow(1-beta)*d.pow(beta),
                                           rtol=1e-14, atol=1e-14)

    def test_common_probability_scaling_gives_epsilon_power_beta(self):
        student = torch.tensor([-1., -2., -7.], dtype=torch.float64)
        teacher = torch.tensor([-2., -1.5, -8.], dtype=torch.float64)
        log_epsilon = -1000.
        for beta in BETAS:
            got = _log_scores(student+log_epsilon, teacher+log_epsilon, beta)
            expected = _log_scores(student, teacher, beta)+beta*log_epsilon
            torch.testing.assert_close(got, expected, rtol=1e-14, atol=1e-12)

    def test_equal_and_adjacent_logp_have_defined_scores(self):
        student = torch.tensor([-.4, -5., 0.], dtype=torch.float64)
        adjacent = torch.nextafter(student, torch.full_like(student, -torch.inf))
        for beta in BETAS:
            self.assertTrue(bool(torch.isneginf(_log_scores(student, student, beta)).all()))
            scores = _log_scores(student, adjacent, beta)
            self.assertTrue(bool(torch.isfinite(scores).all()))
            # As logp and logq approach one another, L approaches p.
            log_r = (adjacent-student).abs().log()
            expected = log_r+beta*student
            torch.testing.assert_close(scores, expected, rtol=1e-14, atol=1e-12)

    def test_extreme_tail_retains_order_without_exp_underflow(self):
        student = torch.tensor([[-10000., -9990., -12000.]], dtype=torch.float64)
        teacher = student-2
        self.assertTrue(bool((student.exp()==0).all()))
        valid = torch.ones_like(student, dtype=torch.bool)
        for beta in (.25, .5, .75, 1., 1.5):
            selected, _ = select_score_tokens(student, teacher, valid, .2, beta)
            self.assertEqual(selected.nonzero().tolist(), [[0, 1]])
            self.assertTrue(bool(torch.isfinite(_log_scores(student, teacher, beta)).all()))
        reward_selected, _ = select_score_tokens(student, teacher, valid, .2, 0.)
        self.assertEqual(reward_selected.nonzero().tolist(), [[0, 0]])

    def test_ties_rounding_and_nonempty_minimum_are_per_response(self):
        valid = torch.arange(7).unsqueeze(0) < torch.tensor([5, 7, 1, 0]).unsqueeze(1)
        student = -torch.ones(4, 7)
        teacher = student.clone()  # every score is zero, including endpoints
        for beta in BETAS:
            chosen, _ = select_score_tokens(student, teacher, valid, .5, beta)
            self.assertEqual(chosen.sum(1).tolist(), [2, 4, 1, 0])
            self.assertEqual(chosen[0].nonzero().flatten().tolist(), [0, 1])
        sparse = torch.tensor([[False, True, False, True, True]])
        chosen, _ = select_score_tokens(student[:1, :5], teacher[:1, :5], sparse, .5, 1.5)
        self.assertEqual(chosen.nonzero().tolist(), [[0, 1], [0, 3]])

    def test_low_low_is_diagnostic_and_never_skips_a_response(self):
        student = torch.tensor([[-6., -8.], [-2., -3.]])
        teacher = torch.tensor([[-9., -9.], [-4., -4.]])
        valid = torch.ones_like(student, dtype=torch.bool)
        for beta in BETAS:
            chosen, stats = select_score_tokens(student, teacher, valid, .5, beta)
            self.assertEqual(chosen.sum(1).tolist(), [1, 1])
            self.assertEqual(stats["valid"], 4)
            self.assertEqual(stats["selected"], 2)
            self.assertEqual(stats["low_low_valid"], 2)
            self.assertEqual(stats["low_low_selected"], 1)
            self.assertEqual(stats["r_abs_total"], 7.)
            self.assertEqual(stats["r_abs_low_total"], 4.)
            self.assertEqual(sum(stats[k] for k in stats if k.startswith("skipped_")), 0)
            changed, _ = select_score_tokens(student, teacher, valid, .5, beta,
                                             probability_threshold=.5)
            self.assertTrue(torch.equal(chosen, changed))

    def test_invalid_positions_and_empty_shapes_are_ignored(self):
        student = torch.tensor([[math.nan, -1., math.inf, -2., 99.]])
        teacher = torch.tensor([[-1000., -2., -1000., -4., math.nan]])
        valid = torch.tensor([[False, True, False, True, False]])
        selected, stats = select_score_tokens(student, teacher, valid, .5, 0.)
        self.assertEqual(selected.nonzero().tolist(), [[0, 3]])
        self.assertEqual(stats["valid"], 2)
        self.assertEqual(stats["r_abs_total"], 3.)
        for shape in ((2, 0), (0, 3), (2, 3)):
            invalid = torch.full(shape, math.nan)
            chosen, stats = select_score_tokens(
                invalid, invalid, torch.zeros(shape, dtype=torch.bool), score_beta=1.5)
            self.assertEqual(tuple(chosen.shape), shape)
            self.assertEqual(sum(stats.values()), 0)

    def test_invalid_inputs_stop(self):
        student = torch.tensor([[-1.]])
        valid = torch.ones_like(student)
        for beta in (math.nan, math.inf, -.1, True):
            with self.assertRaises(ValueError):
                select_score_tokens(student, student, valid, score_beta=beta)
        for ratio in (0., 1.1, math.nan):
            with self.assertRaises(ValueError):
                select_score_tokens(student, student, valid, keep_ratio=ratio)
        for threshold in (0., 1.1, math.inf):
            with self.assertRaises(ValueError):
                select_score_tokens(student, student, valid, probability_threshold=threshold)
        for value in (math.nan, math.inf, -math.inf, .01):
            for which in ("student", "teacher"):
                a, b = student.clone(), student.clone()
                (a if which=="student" else b)[0, 0] = value
                with self.assertRaisesRegex(ValueError, which+"_logp"):
                    select_score_tokens(a, b, valid)
        for mask in (torch.tensor([[.5]]), torch.tensor([[math.nan]]), torch.ones(1, 2)):
            with self.assertRaises(ValueError):
                select_score_tokens(student, student, mask)
        with self.assertRaises(ValueError):
            select_score_tokens(student.flatten(), student.flatten(), valid.flatten())
        with self.assertRaises(ValueError):
            select_score_tokens(student.long(), student, valid)

    def test_original_loss_tensor_keeps_its_gradient(self):
        student = torch.tensor([[-1., -2., -3.]], requires_grad=True)
        teacher = torch.tensor([[-2., -4., -6.]], requires_grad=True)
        selected, _ = select_score_tokens(student, teacher, torch.ones_like(student), .5, 0.)
        (student*selected).sum().backward()
        self.assertEqual(student.grad.tolist(), [[0., 1., 1.]])
        self.assertIsNone(teacher.grad)
        self.assertFalse(selected.requires_grad)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
