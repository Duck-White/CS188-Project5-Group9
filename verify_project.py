"""Supplementary checks for the project, using the unmodified Berkeley data."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import backend
import models
import train


def check_implementations():
    perceptron = models.PerceptronModel(2)
    assert perceptron.get_prediction(torch.zeros(2, dtype=torch.float64)) == 1
    assert perceptron(torch.ones(1, 2, dtype=torch.float64)).shape == (1,)

    matrix = torch.arange(15, dtype=torch.float64).reshape(3, 5)
    kernel = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
    batch = torch.stack((matrix, matrix + 1)).requires_grad_()
    kernel.requires_grad_()
    result = models.Convolve(batch, kernel)
    expected = torch.tensor([[100, 121, 142], [205, 226, 247]], dtype=torch.float64)
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(result[1], expected + 21)
    result.sum().backward()
    assert batch.grad is not None and kernel.grad is not None
    assert train.train_digitconvolution is train.Train_DigitConvolution

    rnn = models.LanguageIDModel().eval()
    characters = torch.eye(rnn.num_chars)[:6].reshape(3, 2, rnn.num_chars)
    padding = torch.zeros(4, 2, rnn.num_chars)
    with torch.inference_mode():
        torch.testing.assert_close(rnn(characters), rnn(torch.cat((characters, padding))))

    attention = models.Attention(layer_size=8, block_size=12).eval()
    sequence = torch.randn(3, 7, 8)
    updated = sequence.clone()
    updated[:, 4:] += 10
    with torch.inference_mode():
        full = attention(sequence)
        assert full.shape == sequence.shape
        torch.testing.assert_close(full[:, :4], attention(updated)[:, :4])
        torch.testing.assert_close(full[:, :4], attention(sequence[:, :4]))
    print("PASS: Q1 score shape, Q4 zero padding, Q5 batched convolution and gradients, Q6 causal prefix")


def accuracy(model, images, labels):
    inputs = torch.from_numpy(np.ascontiguousarray(images)).float()
    targets = torch.from_numpy(np.ascontiguousarray(labels))
    model.eval()
    with torch.inference_mode():
        correct = sum(
            (model(batch).argmax(dim=1) == target).sum().item()
            for batch, target in zip(inputs.split(128), targets.split(128))
        )
    return correct / len(targets)


def evaluate_q5():
    model = models.DigitConvolutionalModel()
    dataset = backend.DigitClassificationDataset2(model)
    train.Train_DigitConvolution(model, dataset)
    training_indices_used_for_grading = np.arange(1, 100, 2)
    assert np.array_equal(dataset.x[training_indices_used_for_grading], dataset.test_images)
    assert np.array_equal(dataset.y.argmax(axis=1)[training_indices_used_for_grading], dataset.test_labels)
    official = accuracy(model, dataset.test_images, dataset.test_labels)
    with np.load(backend.get_data_path("mnist.npz")) as mnist:
        images = mnist["test_images"][1::2]
        labels = mnist["test_labels"][1::2]
    independent = accuracy(model, images, labels)
    print(f"Q5 starter subset: {official:.2%} ({len(dataset.test_labels)} rows, all from Q5 train)")
    print(f"Q5 independent MNIST: {independent:.2%} ({len(labels)} rows from the MNIST test split)")


def export_q3(output_directory, seed):
    model = models.DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model)
    train.train_digitclassifier(model, dataset)
    dev_score = accuracy(model, dataset.dev_images, dataset.dev_labels)
    test_score = accuracy(model, dataset.test_images, dataset.test_labels)
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_directory / "q3_mnist.pt")
    metadata = {
        "mode": "MNIST Q3",
        "source": "DigitClassificationModel và train_digitclassifier của Nhóm 9",
        "dataset": "60.000 ảnh train, 5.000 ảnh dev và 5.000 ảnh test",
        "seed": seed,
        "validation_accuracy": dev_score,
        "test_accuracy": test_score,
        "note": "Checkpoint demo được huấn luyện trong một lần chạy Q3 riêng.",
    }
    (output_directory / "q3_mnist.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Q3 demo checkpoint: dev = {dev_score:.2%}, test = {test_score:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-demo", type=Path, help="Write standalone MNIST demo weights")
    options = parser.parse_args()
    torch.set_num_threads(1)
    backend.use_graphics = False
    print(f"Seed {options.seed} | torch {torch.__version__} | numpy {np.__version__}")
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if options.export_demo is not None:
        export_q3(options.export_demo, options.seed)
    else:
        evaluate_q5()
        check_implementations()


if __name__ == "__main__":
    main()
