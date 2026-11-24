"""The checks a Sol job runs before it starts vLLM."""
import unittest

from contracts.io import load_conversations, load_variants
from models.sol.preflight import CONVERSATIONS, check_gpu, check_variants


class GpuCheck(unittest.TestCase):
    # Lines as nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits prints them.
    def test_gpus_that_can_run_the_job(self):
        for line in ("NVIDIA A100-SXM4-80GB, 81920, 8.0", "NVIDIA A100-PCIE-40GB, 40960, 8.0",
                     "NVIDIA H100 80GB HBM3, 81559, 9.0"):
            self.assertEqual(check_gpu(line)[1], [], line)

    def test_gpus_that_cannot(self):
        for line, reason in (("NVIDIA A30, 24576, 8.0", "MB"),                  # bfloat16 yes, memory too small
                             ("Tesla V100-SXM2-32GB, 32768, 7.0", "bfloat16"),  # no bfloat16
                             ("", "no GPU visible")):
            problems = check_gpu(line)[1]
            self.assertTrue(problems and reason in " ".join(problems), line)


class VariantsCheck(unittest.TestCase):
    def test_the_dataset_variants_pass(self):
        variants = load_variants("dataset/data/variants.jsonl")
        self.assertEqual(check_variants(variants, load_conversations(CONVERSATIONS)), [])

    def test_a_duplicate_is_caught(self):
        variants = load_variants("dataset/data/variants.jsonl")[:2]
        problems = check_variants(variants + variants[:1], load_conversations(CONVERSATIONS))
        self.assertTrue(any("appears 2 times" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
