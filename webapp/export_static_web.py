"""Export TRAINED PyTorch checkpoint(s) to JSON for static Vercel inference.

Use after running 00_HUAN_LUYEN_TAT_CA.bat. Does not invent weights.
Q2 is trained on 200 demo sine points; Q6 is an untrained deterministic layer,
matching the examples provided in the local FastAPI endpoints.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

WEB=Path(__file__).resolve().parent.parent
ROOT=WEB.parent/'02_Ma_nguon' if (WEB.parent/'02_Ma_nguon').is_dir() else WEB
SITE=WEB if ROOT != WEB else WEB/'docs'
sys.path.insert(0,str(ROOT))
import models
import train


def tensor_json(model, question, **meta):
    state=model.state_dict()
    return {'format':'cs188-p5-tensor-v1','question':question,'tensors':{
        key:{'shape':list(value.shape),'data':value.detach().float().reshape(-1).tolist()}
        for key,value in state.items() if value.is_floating_point()
    }, **meta}


def main():
    torch.set_num_threads(1)
    target=SITE/'models-public'
    target.mkdir(exist_ok=True)
    mapping={'q3':models.DigitClassificationModel,'q4':models.LanguageIDModel,'q5':models.DigitConvolutionalModel}
    missing=[]
    for key,klass in mapping.items():
        checkpoint=WEB/'checkpoints'/f'{key}.pt'
        if not checkpoint.exists():
            missing.append(key)
            continue
        model=klass()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        meta={}
        if key=='q4':
            alpha=WEB/'checkpoints'/'q4_chars.json'
            if not alpha.exists():
                missing.append(key+'-chars')
                continue
            letters=json.loads(alpha.read_text(encoding='utf8'))
            if len(letters)!=47:
                raise ValueError('q4_chars.json phải có đúng 47 ký tự')
            meta['alphabet']=letters
        (target/f'{key}.json').write_text(json.dumps(tensor_json(model,key,**meta),ensure_ascii=False,separators=(',',':')),encoding='utf8')
        print('Xuất:',target/f'{key}.json')
    # Q2 is an explicitly DEMO regression trained with group model and train code.
    from webapp.server import TinyDataset
    import math
    torch.manual_seed(188)
    q2=models.RegressionModel()
    xs=torch.linspace(-2*math.pi,2*math.pi,200).view(-1,1)
    ys=torch.sin(xs)
    train.train_regression(q2,TinyDataset(xs,ys))
    (target/'q2.json').write_text(json.dumps(tensor_json(q2,'q2'),separators=(',',':')),encoding='utf8')
    # Q6 causal attention is a demo of math; it does not have trained weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(188)
        q6=models.Attention(layer_size=4,block_size=6)
    (target/'q6.json').write_text(json.dumps(tensor_json(q6,'q6'),separators=(',',':')),encoding='utf8')
    if missing:
        print('CHƯA XUẤT do thiếu checkpoint:',', '.join(missing))
        raise SystemExit(1)
    release=SITE/'Vercel_Release'
    (release/'models-public').mkdir(parents=True,exist_ok=True)
    for name in ('index.html','browser-infer.js'):
        shutil.copy2(SITE/name,release/name)
    for file in target.glob('*.json'):
        shutil.copy2(file,release/'models-public'/file.name)
    print('Da tao folder Vercel_Release. Tai folder len Vercel de Q2-Q6 chay tren browser.')


if __name__=='__main__':
    main()