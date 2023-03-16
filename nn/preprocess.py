# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool], seed: int = 1) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels
        seed: int
            Default = 1, seed for random sampling of classes

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Set a random seed for reproducability
    np.random.seed(seed)

    seq_arr = np.array(seqs)
    lab_arr = np.array(labels)

    # Get the postitive and negative sequences, where labels are True and False, respectively
    seq_pos = seq_arr[lab_arr == True]
    seq_neg = seq_arr[lab_arr == False]

    # If more positive sequences than negative sequences, oversample the negatives
    if len(seq_pos) > len(seq_neg):
        os_neg = seq_neg[np.random.choice(len(seq_neg), len(seq_pos), replace = True)]
        # Create list of new sampled sequences and labels
        sampled_seqs = list(np.concatenate((seq_pos, os_neg), axis = None))
        sampled_labs = list([True] * len(seq_pos) + [False] * len(os_neg))

    # If more negative sequences that positive sequences, oversample the positives
    elif len(seq_pos) < len(seq_neg):
        os_pos = seq_pos[np.random.choice(len(seq_pos), len(seq_neg), replace = True)]
        # Create list of new sampled sequences and labels
        sampled_seqs = list(np.concatenate((os_pos, seq_neg), axis = None))
        sampled_labs = list([True] * len(os_pos) + [False] * len(seq_neg))
        
    # If they are balanced, then return original list of seqs and labels
    elif len(seq_pos) == len(seq_neg):
        sampled_seqs = seqs
        sampled_labs = labels

    return sampled_seqs, sampled_labs

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Array of coded sequences
    codes = {'A':[1,0,0,0],
             'T':[0,1,0,0],
             'C':[0,0,1,0],
             'G':[0,0,0,1]}
    
    # Translate the input seq_arr
    encoding = []
    for seq in seq_arr:
        e = np.array([codes[i] for i in seq])
        encoding.append(e.flatten())
    encoding = np.array(encoding)

    return(encoding)