import unittest
import torch
from mt.data import make_splits, normalize
from mt.basic import BasicTransformer, Vocabulary, choose_token, PAD, BOS
from mt.experiment import score, make_prompt


class DataTests(unittest.TestCase):
    def test_deduplication_and_disjoint_splits(self):
        config = {"seed": 9, "train_size": 2, "validation_size": 1, "test_size": 1, "min_words": 1, "max_words": 10}
        lines = ["One\tUn", "ONE\tUne", "Two\tDeux", "Three\tTrois", "Four\tQuatre", "Else\tDeux"]
        splits, counts = make_splits(lines, config)
        self.assertEqual(counts["duplicates"], 2)
        sources = [{r["source"].casefold() for r in rows} for rows in splits.values()]
        self.assertFalse(sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2])
        self.assertEqual(splits, make_splits(lines, config)[0])

    def test_unicode_and_vocabulary_leakage(self):
        self.assertEqual(normalize("  cafe\u0301\u202f!  "), "café !")
        vocab = Vocabulary.fit(["Hello world"], 100)
        self.assertEqual(vocab.encode("unseen", 10)[1], 3)


class ModelTests(unittest.TestCase):
    def test_decoder_cannot_see_future_tokens(self):
        torch.manual_seed(1)
        model = BasicTransformer(20, 20, {"d_model": 16, "nhead": 2, "layers": 1, "dim_feedforward": 32, "dropout": 0.}).eval()
        source = torch.tensor([[1, 7, 2]])
        a, b = torch.tensor([[1, 4, 5]]), torch.tensor([[1, 4, 9]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(source, a)[:, :2], model(source, b)[:, :2], atol=1e-6))

    def test_optimizer_updates_and_loss_falls(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        model = BasicTransformer(12, 12, {"d_model": 16, "nhead": 2, "layers": 1, "dim_feedforward": 32, "dropout": 0.})
        source, target = torch.tensor([[1, 5, 2]]), torch.tensor([[1, 7]])
        labels = torch.tensor([[7, 2]])
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(source, target).flatten(0, 1), labels.flatten())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0]/3)

    def test_sampling_excludes_specials(self):
        logits = torch.tensor([[100., 90., 1., 2., 3.]])
        self.assertEqual(choose_token(logits, 0, .9).item(), 4)
        for _ in range(10):
            self.assertNotIn(choose_token(logits, .7, .9).item(), (PAD, BOS))

    def test_metrics_and_alignment(self):
        result = score(["Bonjour à tout le monde !"], ["Bonjour à tout le monde !"])
        self.assertAlmostEqual(result["bleu"], 100.)
        self.assertAlmostEqual(result["chrf"], 100.)
        with self.assertRaises(ValueError):
            score(["a"], ["a", "b"])

    def test_prompt_contains_no_test_reference(self):
        prompt = make_prompt("Test source", [{"source": "Train source", "target": "Train target"}])
        self.assertIn("Train target", prompt)
        self.assertIn("Test source", prompt)
        self.assertTrue(prompt.endswith("French:"))


if __name__ == "__main__":
    unittest.main()
