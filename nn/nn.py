# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.
        verbose: bool
            Print messages if True. Only do this with the testing and for small epochs/layers unless you wanna blow up your computer

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str,
        verbose : bool = False
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()
        self._verbose = verbose
        self.cache = []

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Apply linear transformation
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T

        # Apply the activation function depending on activation type
        if activation.lower() == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif activation.lower() == "relu":
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError("Invalid activation function")

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Initialize cache 
        cache = {"A0" : X}
        A_prev = X
        # Iterate through the arch
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1

            # if layer_idx == 1:
            #     A_prev = X
            # else:
            #     A_prev = A_curr

            _W_curr = self._param_dict['W' + str(layer_idx)]
            _b_curr = self._param_dict['b' + str(layer_idx)]
            _activation = layer["activation"]
            if self._verbose:
                print("Layer index: " + str(layer_idx))
                print("Shape _W :" + str(_W_curr.shape))
                print("Shape _b :" + str(_b_curr.shape))
                print("Shape _A_prev: " + str(A_prev.shape))

            # Run _single_forward
            A_curr, Z_curr = self._single_forward(W_curr = _W_curr,
                                                  b_curr = _b_curr,
                                                  A_prev = A_prev,
                                                  activation = _activation)
            if self._verbose:
                print("Shape Z_curr :" + str(Z_curr.shape))
                print("Shape A_curr: " + str(A_curr.shape))
            # Cache results
            cache['Z' + str(layer_idx)] = Z_curr
            cache['A' + str(layer_idx)] = A_curr
            A_prev = A_curr

        output = A_prev
        self.cache = cache
        return output, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # Activation specific backprop, calculating dA (wrt current Z layer)
        if activation_curr.lower() == "sigmoid":
            bp = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr.lower() == "relu":
            bp = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError("Invalid activation function")
        
        if self._verbose:
            print("Shape bp: " + str(bp.shape))

        # Calculate dA, dW, db
        dA_prev = np.dot(bp, W_curr)
        dW_curr = np.dot(bp.T, A_prev)
        db_curr = np.sum(bp, axis = 0)
        db_curr = db_curr.reshape(b_curr.shape)

        return dA_prev, dW_curr, db_curr



    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        # Initialize grad_dict
        grad_dict = {}

        # Calculate dA of the current layer
        if self._loss_func.lower() == "mse":
            _dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func.lower() == "bce":
            _dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else: 
            raise ValueError("Invalid loss function")

        for idx, layer in reversed(list(enumerate(self.arch))):
            # Get current layer index
            layer_idx = idx + 1
            
            # Get parameters for _single_backprop
            _W_curr = self._param_dict['W' + str(layer_idx)]
            _b_curr = self._param_dict['b' + str(layer_idx)]
            _Z_curr = cache['Z' + str(layer_idx)]
            _A_prev = cache['A' + str(idx)]
            _activation = layer['activation']
            if self._verbose:
                print("Layer index: " + str(layer_idx))
                print("Shape _W_curr :" + str(_W_curr.shape))
                print("Shape _b_curr :" + str(_b_curr.shape))
                print("Shape _Z_curr: " + str(_Z_curr.shape))
                print("Shape _A_prev:" + str(_A_prev.shape))
                print("Shape _dA: " + str(_dA_curr.shape))
            

            # Run single backprop
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr = _W_curr,
                                                              b_curr = _b_curr,
                                                              Z_curr = _Z_curr,
                                                              A_prev = _A_prev,
                                                              dA_curr = _dA_curr,
                                                              activation_curr = _activation)
        
            # Populate grad_dict with results from single backprop
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr
            # Update the dA
            _dA_curr = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            # Update param by appropriate gradient * lr
            self._param_dict["W" + str(layer_idx)] -= grad_dict["dW" + str(layer_idx)] * self._lr 
            self._param_dict['b' + str(layer_idx)] -= grad_dict["db" + str(layer_idx)] * self._lr 


    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Initialize per epochs
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # Calculate the number of batches
        num_batches = np.ceil(X_train.shape[0] / self._batch_size)

        # Iterate through epochs
        for epoch in range(self._epochs):

            # Shuffle training data
            shuffle = np.random.permutation(len(y_train))
            _X_train = X_train[shuffle]
            _y_train = y_train[shuffle]

            # Create batches
            X_batch = np.array_split(_X_train, num_batches)
            y_batch = np.array_split(_y_train, num_batches)

            # Initialize batch_loss
            batch_train_loss = []
            batch_val_loss = []

            for _X_batch, _y_batch in zip(X_batch, y_batch):

                # Run forward
                if self._verbose:
                    print("Forward:")
                output, cache = self.forward(_X_batch)
                if self._loss_func.lower() == "mse":
                    batch_train_loss.append(self._mean_squared_error(_y_batch, output))
                elif self._loss_func.lower() == "bce":
                    batch_train_loss.append(self._binary_cross_entropy(_y_batch, output))
                else: 
                    raise ValueError("Invalid loss function")
                    
                # Backpropogate, passing in the true labels y (-y_batch)
                if self._verbose:
                    print("Backprop")
                grad_dict = self.backprop(_y_batch, output, cache)
                self._update_params(grad_dict)

                # Calculate average training loss
                per_epoch_loss_train.append(np.mean(batch_train_loss))

                # Compute validation loss
                pred = self.predict(X_val)
                if self._loss_func.lower() == "mse":
                    batch_val_loss.append(self._mean_squared_error(y_val, pred))
                elif self._loss_func.lower() == "bce":
                    batch_val_loss.append(self._binary_cross_entropy(y_val, pred))
                
                # Calculate average validation loss
                per_epoch_loss_val.append(np.mean(batch_val_loss))

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # StraightFORWARD (hah pun intended)
        y_hat, _ = self.forward(X)
        
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / ( 1 + np.exp(-Z))
        
        return(nl_transform)

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(Z,0)
        
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = dA * (Z > 0).astype(int)
        
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Should implement epsilon value here, but too lazy, instead using np.clip
        # Prevent NaN and Inf values
        y_hat = np.clip(y_hat, 0.00001, 0.99999)
        # Calculate loss
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        
        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Should implement epsilon value here, but too lazy, instead using np.clip
        # Prevent NaN and Inf values
        y_hat = np.clip(y_hat, 0.00001, 0.99999)
        dA = (((1 - y) / (1 - y_hat)) - (y / y_hat)) / len(y)
        
        if self._verbose:
            print("_BCE_BP dA shape: " + str(dA.shape))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.mean(np.square(y - y_hat))
        
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = (-2 * (y - y_hat)) / len(y)
        if self._verbose:
            print("_MSE_BP dA shape: " + str(dA.shape))
        return dA