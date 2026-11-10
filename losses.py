from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module
import torch
from torch.nn import Parameter, Linear
from torch import tensor, ones, matmul
from torch.nn.functional import relu, cross_entropy, softmax
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset, LanguageIDDataset

class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        return self.w

    def run(self, x):
        return self(x)

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        return matmul(x, self.w.T)

    def get_prediction(self, x):
        "*** YOUR CODE HERE ***"
        return 1 if self(x).item() >= 0 else -1

class RegressionModel(Module):
    def __init__(self):
        super().__init__()
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(1, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 1)

    def run(self, x):
        return self(x)

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)

class DigitClassificationModel(Module):
    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)

    def run(self, x):
        return self(x)

    def forward(self, x):# =====================================================================
# SCRIPT TỰ ĐỘNG HOÀN THÀNH TOÀN BỘ TASK CỦA CƯỜNG (GĐ1 -> GĐ4) A-Z
# BẢN CẬP NHẬT: CODE SẠCH, KHÔNG CHỨA LỖI ẢO Q5, Q6
# =====================================================================
$ErrorActionPreference = "Continue"

# 1. Xác định đường dẫn Python 3.12
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "C:\Program Files\Python312\python.exe" }
if (-not (Test-Path $py)) { $py = "python" }

# 2. Di chuyển vào thư mục dự án và đảm bảo đứng đúng branch Feature-Q1
$projDir = "D:\TTNT\CS188-Project5-Group9"
$outDir  = "D:\TTNT\BanGiao_Cuong_Full"
New-Item -ItemType Directory -Force $projDir | Out-Null
New-Item -ItemType Directory -Force $outDir  | Out-Null
Set-Location $projDir

if (-not (Test-Path ".git")) {
    git init
    git remote add origin https://github.com/Duck-White/CS188-Project5-Group9.git
}
git fetch origin
git checkout -B Feature-Q1

# 3. Ghi file losses.py chuẩn (Q2, Q3, Q4, Q5)
@'
from torch.nn.functional import mse_loss, cross_entropy

def regression_loss(y_pred, y):
    "*** YOUR CODE HERE ***"
    return mse_loss(y_pred, y)

def digitclassifier_loss(y_pred, y):
    "*** YOUR CODE HERE ***"
    return cross_entropy(y_pred, y)

def languageid_loss(y_pred, y):
    "*** YOUR CODE HERE ***"
    return cross_entropy(y_pred, y)

def digitconv_loss(y_pred, y):
    "*** YOUR CODE HERE ***"
    return cross_entropy(y_pred, y)
