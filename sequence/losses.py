"""
Loss functions for recommender models.

The pointwise, BPR, and hinge losses are a good fit for
implicit feedback models trained through negative sampling.

The regression and Poisson losses are used for explicit feedback
models.
"""
import tensorflow as tf
import tensorflow.keras as keras


def pointwise_loss(positive_predictions, negative_predictions, mask=None):
    """
    Logistic loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    positives_loss = - tf.math.log_sigmoid(positive_predictions)
    negatives_loss = - tf.math.log(1 - tf.math.sigmoid(negative_predictions))

    loss = (positives_loss + negatives_loss)

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        loss = loss * mask

    return tf.math.reduce_mean(loss)


def bpr_loss(positive_predictions, negative_predictions, sample=1, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """
    if sample == 1:
        loss = -tf.math.log_sigmoid(positive_predictions - negative_predictions)
    else:
        pass
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        loss = loss * mask

    return tf.math.reduce_mean(loss)


def hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Hinge pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    #pos_score = tf.math.sigmoid(positive_predictions)
    #neg_score = tf.math.sigmoid(positive_predictions)
    pos_score = positive_predictions
    neg_score = negative_predictions
    loss = tf.math.maximum(neg_score - pos_score + 1, 0.0)

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        loss = loss * mask

    return tf.math.reduce_mean(loss)


def adaptive_hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closes to violating the
    ranking implicit in the pattern of user interactions.

    Approximates the idea of weighted approximate-rank pairwise loss
    introduced in [2]_

    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Iterable of tensors containing predictions for sampled negative items.
        More tensors increase the likelihood of finding ranking-violating
        pairs, but risk overfitting.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    References
    ----------
    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011.
    """
    # reduce_max dimension: negtive num dimension
    highest_negative_predictions, _ = tf.math.reduce_max(negative_predictions, 0)
    return hinge_loss(positive_predictions, highest_negative_predictions.squeeze(), mask=mask)


def regression_loss(observed_ratings, predicted_ratings):
    """
    Regression loss.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    return tf.math.reduce_mean(((observed_ratings - predicted_ratings) ** 2))


def logistic_loss(observed_ratings, predicted_ratings):
    """
    Logistic loss for explicit data.

    Parameters
    ----------

    observed_ratings: tensor
        Tensor containing observed ratings which
        should be +1 or -1 for this loss function.
    predicted_ratings: tensor
        Tensor containing rating predictions.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """
    pass
