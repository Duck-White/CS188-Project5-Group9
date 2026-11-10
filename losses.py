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

def digitconvolution_Loss(y_pred, y):
    "*** YOUR CODE HERE ***"
    return cross_entropy(y_pred, y)

digitconv_loss = digitconvolution_Loss
