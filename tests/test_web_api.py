import math
from fastapi.testclient import TestClient
from webapp.server import app

client=TestClient(app)

def test_home_and_assets():
    response=client.get('/')
    assert response.status_code==200
    assert 'Perceptron' in response.text and 'Causal Attention' in response.text
    assert 'Question 7' not in response.text
    assert client.get('/assets/app.js').status_code==200
    assert client.get('/assets/style.css').status_code==200


def test_status_no_weights_claim():
    data=client.get('/api/status').json()
    assert data['backend'] is True
    assert data['q3'] is False and data['q4'] is False and data['q5'] is False


def test_perceptron_train_real_model():
    r=client.post('/api/q1',json={})
    assert r.status_code==200,r.text
    data=r.json()
    assert math.isclose(data['accuracy'],1)
    assert len(data['samples'])==12
    assert len(data['trained_weights'])==3


def test_classification_refuses_missing_checkpoint():
    p=[0.0]*784
    r=client.post('/api/q3',json={'pixels':p})
    assert r.status_code==409 and 'q3.pt' in r.json()['detail']
    r=client.post('/api/q5/predict',json={'pixels':p})
    assert r.status_code==409
    r=client.post('/api/q4',json={'word':'hello'})
    assert r.status_code==409


def test_convolution_matches_known_valid_corr():
    image=[[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]
    kernel=[[1,0,-1],[1,0,-1],[1,0,-1]]
    response=client.post('/api/q5/convolve',json={'image':image,'kernel':kernel})
    assert response.status_code==200,response.text
    d=response.json()
    assert d['shape']==[3,3]
    import torch
    expected=torch.nn.functional.conv2d(torch.tensor(image,dtype=torch.float32)[None,None,:,:],torch.tensor(kernel,dtype=torch.float32)[None,None,:,:]).squeeze().tolist()
    assert d['output']==expected


def test_attention_strict_causal():
    tokens=[[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    response=client.post('/api/q6',json={'tokens':tokens})
    assert response.status_code==200,response.text
    d=response.json()
    assert len(d['attention'])==4
    assert len(d['output'])==4 and len(d['output'][0])==4
    for i,row in enumerate(d['attention']):
        assert all(v==0 for v in row[i+1:])
        assert abs(sum(row)-1)<1e-5


def test_reject_invalid_inputs():
    assert client.post('/api/q3',json={'pixels':[1.]*784+[0.]}).status_code==422
    assert client.post('/api/q5/convolve',json={'image':[[1],[2,3]],'kernel':[[1]]}).status_code==422
    assert client.post('/api/q6',json={'tokens':[[1,2],[3,4]]}).status_code==422


def test_sine_demo_training():
    response=client.post('/api/q2/train',json={})
    assert response.status_code==200,response.text
    d=response.json()
    assert d['mse']<0.02,d
    curve=client.get('/api/q2/curve')
    assert curve.status_code==200
    assert len(curve.json()['prediction'])==121