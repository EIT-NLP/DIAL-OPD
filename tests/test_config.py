"""Guard the DIAL objective against accidental legacy configuration changes."""
import unittest
from dial_opd.config import DIALConfig


class DIALConfigTests(unittest.TestCase):
    def config(self, **kwargs):
        return DIALConfig(use_cpu=True, bf16=False, report_to='none', **kwargs)

    def test_defaults_preserve_the_sampled_opd_objective(self):
        config = self.config()
        self.assertEqual((config.beta, config.lmbda, config.num_generations), (1., 1., 1))

    def test_alternative_objectives_are_rejected(self):
        for name, value in {'beta': .5, 'lmbda': .5, 'num_generations': 2,
                            'use_uld_loss': True, 'seq_kd': True,
                            'use_liger_kernel': True}.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'DIAL requires'):
                self.config(**{name:value})

    def test_removed_experiment_options_are_not_silently_accepted(self):
        for name in ('reward_mode', 'power_reward_alpha', 'reward_normalization',
                     'selective_filter_enabled', 'token_type_mask_mode'):
            with self.subTest(name=name), self.assertRaises(TypeError):
                self.config(**{name:True})

    def test_eval_sample_counts_are_consistent(self):
        for n,k in ((0,1),(1,2),(2,0)):
            with self.subTest(n=n,k=k), self.assertRaisesRegex(ValueError,'eval_k'):
                self.config(eval_n=n,eval_k=k)


if __name__ == '__main__':
    unittest.main()
