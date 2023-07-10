# %% 
import numpy as np
import librosa
import pandas as pd

from numba import jit

# def lpc(y: np.ndarray, *, order: int, axis: int = -1) -> np.ndarray:
def lpc(y, *, order, axis= -1):
    """Linear Prediction Coefficients via Burg's method

    This function applies Burg's method to estimate coefficients of a linear
    filter on ``y`` of order ``order``.  Burg's method is an extension to the
    Yule-Walker approach, which are both sometimes referred to as LPC parameter
    estimation by autocorrelation.

    It follows the description and implementation approach described in the
    introduction by Marple. [#]_  N.B. This paper describes a different method, which
    is not implemented here, but has been chosen for its clear explanation of
    Burg's technique in its introduction.

    .. [#] Larry Marple.
           A New Autoregressive Spectrum Analysis Algorithm.
           IEEE Transactions on Acoustics, Speech, and Signal Processing
           vol 28, no. 4, 1980.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Time series to fit. Multi-channel is supported..
    order : int > 0
        Order of the linear filter
    axis : int
        Axis along which to compute the coefficients

    Returns
    -------
    a : np.ndarray [shape=(..., order + 1)]
        LP prediction error coefficients, i.e. filter denominator polynomial.
        Note that the length along the specified ``axis`` will be ``order+1``.

    Raises
    ------
    ParameterError
        - If ``y`` is not valid audio as per `librosa.util.valid_audio`
        - If ``order < 1`` or not integer
    FloatingPointError
        - If ``y`` is ill-conditioned

    See Also
    --------
    scipy.signal.lfilter

    Examples
    --------
    Compute LP coefficients of y at order 16 on entire series

    >>> y, sr = librosa.load(librosa.ex('libri1'))
    >>> librosa.lpc(y, order=16)

    Compute LP coefficients, and plot LP estimate of original series

    >>> import matplotlib.pyplot as plt
    >>> import scipy
    >>> y, sr = librosa.load(librosa.ex('libri1'), duration=0.020)
    >>> a = librosa.lpc(y, order=2)
    >>> b = np.hstack([[0], -1 * a[1:]])
    >>> y_hat = scipy.signal.lfilter(b, [1], y)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(y)
    >>> ax.plot(y_hat, linestyle='--')
    >>> ax.legend(['y', 'y_hat'])
    >>> ax.set_title('LP Model Forward Prediction')

    """
    # if not util.is_positive_int(order):
    #     raise ParameterError(f"order={order} must be an integer > 0")

    # util.valid_audio(y, mono=False)

    # Move the lpc axis around front, because numba is silly
    y = y.swapaxes(axis, 0)

    dtype = y.dtype

    shape = list(y.shape)
    shape[0] = order + 1

    ar_coeffs = np.zeros(tuple(shape), dtype=dtype)
    ar_coeffs[0] = 1

    ar_coeffs_prev = ar_coeffs.copy()

    shape[0] = 1
    reflect_coeff = np.zeros(shape, dtype=dtype)
    den = reflect_coeff.copy()

    # epsilon = util.tiny(den)
    epsilon = np.finfo(dtype).tiny # epsilon is the smallest representable number such that 1.0 + epsilon != 1.0

    # Call the helper, and swap the results back to the target axis position
    return np.swapaxes(
        __lpc(y, order, ar_coeffs, ar_coeffs_prev, reflect_coeff, den, epsilon), 0, axis
    )

# @jit(nopython=True, cache=True)  # type: ignore # this is the problem
def __lpc(
    y,
    order,
    ar_coeffs,
    ar_coeffs_prev,
    reflect_coeff,
    den,
    epsilon,
):
#     y: np.ndarray,
#     order: int,
#     ar_coeffs: np.ndarray,
#     ar_coeffs_prev: np.ndarray,
#     reflect_coeff: np.ndarray,
#     den: np.ndarray,
#     epsilon: float,
# ) -> np.ndarray:
    # This implementation follows the description of Burg's algorithm given in
    # section III of Marple's paper referenced in the docstring.
    #
    # We use the Levinson-Durbin recursion to compute AR coefficients for each
    # increasing model order by using those from the last. We maintain two
    # arrays and then flip them each time we increase the model order so that
    # we may use all the coefficients from the previous order while we compute
    # those for the new one. These two arrays hold ar_coeffs for order M and
    # order M-1.  (Corresponding to a_{M,k} and a_{M-1,k} in eqn 5)

    # These two arrays hold the forward and backward prediction error. They
    # correspond to f_{M-1,k} and b_{M-1,k} in eqns 10, 11, 13 and 14 of
    # Marple. First they are used to compute the reflection coefficient at
    # order M from M-1 then are re-used as f_{M,k} and b_{M,k} for each
    # iteration of the below loop
    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    # DEN_{M} from eqn 16 of Marple.
    den[0] = np.sum(fwd_pred_error**2 + bwd_pred_error**2, axis=0)

    for i in range(order):
        # can be removed if we keep the epsilon bias
        # if np.any(den <= 0):
        #    raise FloatingPointError("numerical error, input ill-conditioned?")

        # Eqn 15 of Marple, with fwd_pred_error and bwd_pred_error
        # corresponding to f_{M-1,k+1} and b{M-1,k} and the result as a_{M,M}

        reflect_coeff[0] = np.sum(bwd_pred_error * fwd_pred_error, axis=0)
        reflect_coeff[0] *= -2
        reflect_coeff[0] /= den[0] + epsilon

        # Now we use the reflection coefficient and the AR coefficients from
        # the last model order to compute all of the AR coefficients for the
        # current one.  This is the Levinson-Durbin recursion described in
        # eqn 5.
        # Note 1: We don't have to care about complex conjugates as our signals
        # are all real-valued
        # Note 2: j counts 1..order+1, i-j+1 counts order..0
        # Note 3: The first element of ar_coeffs* is always 1, which copies in
        # the reflection coefficient at the end of the new AR coefficient array
        # after the preceding coefficients

        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            # reflection multiply should be broadcast
            ar_coeffs[j] = (
                ar_coeffs_prev[j] + reflect_coeff[0] * ar_coeffs_prev[i - j + 1]
            )
            print(f'ar_coeffs[j]: {ar_coeffs[j]}')

        # Update the forward and backward prediction errors corresponding to
        # eqns 13 and 14.  We start with f_{M-1,k+1} and b_{M-1,k} and use them
        # to compute f_{M,k} and b_{M,k}
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp
        print(f'fwd_pred_error: {fwd_pred_error}')
        print(f'bwd_pred_error: {bwd_pred_error}')

        # SNIP - we are now done with order M and advance. M-1 <- M

        # Compute DEN_{M} using the recursion from eqn 17.
        #
        # reflect_coeff = a_{M-1,M-1}      (we have advanced M)
        # den =  DEN_{M-1}                 (rhs)
        # bwd_pred_error = b_{M-1,N-M+1}   (we have advanced M)
        # fwd_pred_error = f_{M-1,k}       (we have advanced M)
        # den <- DEN_{M}                   (lhs)
        #

        q = 1.0 - reflect_coeff[0] ** 2
        den[0] = q * den[0] - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        # Shift up forward error.
        #
        # fwd_pred_error <- f_{M-1,k+1}
        # bwd_pred_error <- b_{M-1,k}
        #
        # N.B. We do this after computing the denominator using eqn 17 but
        # before using it in the numerator in eqn 15.
        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs

# %%
# load data
df_data = pd.read_csv('../data_path.csv')

# config
order = 5
sample_rate = 22050 * 2 
sample_rate = 192000
swav, sr = librosa.load(df_data['audio_path'].values[0], sr=sample_rate)
print(f'wave shape: {swav.shape}')

# %%
x = swav.copy()
burg_coeff = lpc(y=x, order=order)
print(f'scratch coeff : {burg_coeff}')
print(f'-------------- AR coefficients from librosa implementation -------------- ')
lib_coeff = librosa.lpc(y=x, order=order)
print(f'lib_coeff: {lib_coeff}')
print(f'with sampling rate: {sr}')
print(f'Is close?: {burg_coeff - lib_coeff}')
# %%

# Why librosa is so slow?
# the problem might come from @jit
# from numba import jit -> compile a python function into native code
# It is about numba
# which implementation is correct?
# %%
