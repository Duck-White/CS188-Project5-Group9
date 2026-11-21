"""Train real Berkeley datasets and export local checkpoints for the web demo.

These checkpoints are not required for Gradescope and are deliberately not
included in the repository ZIP. Run only with the official data/ files present.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import backend
import models
import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", choices=["q3", "q4", "q5"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    backend.use_graphics = False
    choices = {
        "q3": (models.DigitClassificationModel, backend.DigitClassificationDataset, train.train_digitclassifier),
        "q4": (models.LanguageIDModel, backend.LanguageIDDataset, train.train_languageid),
        "q5": (models.DigitConvolutionalModel, backend.DigitClassificationDataset2, train.Train_DigitConvolution),
    }
    model_class, dataset_class, train_fn = choices[args.question]
    model = model_class()
    dataset = dataset_class(model)
    train_fn(model, dataset)
    target = ROOT / 'checkpoints'
    target.mkdir(exist_ok=True)
    torch.save(model.eval().state_dict(), target / (args.question + '.pt'))
    if args.question == 'q4':
        (target/'q4_chars.json').write_text(json.dumps([str(x) for x in dataset.chars.tolist()], ensure_ascii=False), encoding='utf8')
    print('Đã xuất', target / (args.question + '.pt'))
    print('Đây là checkpoint local; hãy chạy autograder.py để đánh giá chính thức.')


if __name__ == "__main__":
    main()