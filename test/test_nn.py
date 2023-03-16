import numpy as np
from nn import nn, preprocess

def simple_nn():
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 1,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")
    return net


# def test_single_forward():
#     pass

# def test_forward():
#     pass

# def test_single_backprop():
#     pass

# def test_predict():
#     pass

def test_binary_cross_entropy():
    # Test with y and y_hat, comapre to manual calculation
    net = simple_nn()
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 1, 1, 0])
    
    bce = net._binary_cross_entropy(y, y_hat)

    assert round(bce, 3) == 5.756

def test_binary_cross_entropy_backprop():
    # Test with y and y_hat, comapre to manual calculation
    net = simple_nn()
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 1, 1, 0])
    dA = net._binary_cross_entropy_backprop(y, y_hat)
    dA_round = [round(i,2) for i in dA]

    assert all(dA_round == np.array([0.25, 25000.0, -0.25, -25000.0]))

def test_mean_squared_error():
    # Test with y and y_hat, comapre to manual calculation
    net = simple_nn()
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 1, 1, 0])

    mse = net._mean_squared_error(y, y_hat)
    
    assert mse == 0.5

def test_mean_squared_error_backprop():
    # Test with y and y_hat, comapre to manual calculation
    net = simple_nn()
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 1, 1, 0])
    
    mse_backprop = net._mean_squared_error_backprop(y, y_hat)
    
    assert all(mse_backprop == np.array([ 0. ,  0.5,  0. , -0.5]))

def test_sample_seqs():
    # Test when there are more positive samples than negative samples
    seqs = ["test1", "test2", "test3", "test4", "test5"]
    labs = [True, False, True, True, False]
    new_seqs, new_labs = preprocess.sample_seqs(seqs, labs)
    
    # Assert that the number of trues is half the total number
    assert sum(new_labs) == len(new_labs)/2
    
    # Ensure that the values that are true are actually ture
    true_seqs = [val for i, val in enumerate(new_seqs) if new_labs[i]]
    true_idx = [seqs.index(i) for i in true_seqs]
    
    assert all([labs[i] for i in true_idx])
    
    # Test when there are more negative samples than positive samples
    seqs = ["test1", "test2", "test3", "test4", "test5"]
    labs = [True, False, True, False, False]
    new_seqs, new_labs = preprocess.sample_seqs(seqs, labs)
    
    # Assert that the number of trues is half the total number
    assert sum(new_labs) == len(new_labs)/2 
    
    # Ensure that the values that are true are actually ture
    true_seqs = [val for i, val in enumerate(new_seqs) if new_labs[i]]
    true_idx = [seqs.index(i) for i in true_seqs]
    
    assert all([labs[i] for i in true_idx])

def test_one_hot_encode_seqs():
    seqs = ["AT",
            "CG"]
    encoded_true = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 1]])
    
    encoded_seq = preprocess.one_hot_encode_seqs(seqs)

    assert np.all(encoded_true == encoded_seq)