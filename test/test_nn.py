import numpy as np
from nn import nn, preprocess

def simple_nn():
    net = nn.NeuralNetwork(nn_arch = [{"input_dim": 4, "output_dim": 2, "activation": "relu"},
                                      {"input_dim": 2, "output_dim": 1, "activation": "relu"}],
                           lr = 0.001,
                           seed = 1,
                           batch_size = 1,
                           epochs = 1,
                           loss_function = "mse")
    return net

def test_single_forward():
    net = simple_nn()

    # Create own inputs
    W_curr = np.array([[10, 20, 30, 10, 20]])
    b_curr = np.array([[1]])
    A_prev = np.array([[1, 2, 3, 2, 1]])

    # Calculate single forward
    A_curr, Z_curr = net._single_forward(W_curr, b_curr, A_prev, "relu")

    # Compare values
    assert np.array_equal(A_curr, np.array([[181]]))
    assert np.array_equal(Z_curr, np.array([[181]]))   

def test_forward():
    net = simple_nn()

    # Create own inputs
    net._param_dict = {"W1": np.array([[1, 2, 3], [1, 2, 3]]),
                      "b1": np.array([[1], [1]]),
                      "W2": np.array([[1, 1]]),
                      "b2": np.array([[1]])}

    # Calculate forward
    output, cache = net.forward(np.array([2, 2, 2]))
    assert output == 27

def test_single_backprop():
    net = simple_nn()

    # Create own inputs
    W_curr = np.array([[1, 2, 3, 4, 5]])
    b_curr = np.array([[1]])
    Z_curr = np.array([[2]])
    A_prev = np.array([[1, 2, 3, 2, 1]])
    dA_curr = np.array([[3]])

    # Calcualte single backprop
    dA_prev, dW_curr, db_curr = net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu")
    assert np.array_equal(dA_prev, np.array([[3, 6, 9, 12, 15]]))
    assert np.array_equal(dW_curr, np.array([[3, 6, 9, 6, 3]]))
    assert np.array_equal(db_curr, np.array([[3]]))

def test_predict():
    net = simple_nn()

    # Create own inputs
    net._param_dict = {"W1": np.array([[1, 2, 3], [1, 2, 3]]),
                      "b1": np.array([[1], [1]]),
                      "W2": np.array([[1, 1]]),
                      "b2": np.array([[1]])}
    
    pred = net.predict(np.array([2, 2, 2,]))
    assert pred == 27

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

