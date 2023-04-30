""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'Simply SDR Loss'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import keras.backend as K
import tensorflow as tf
def modified_SDR_loss(pred, true, eps = 1e-8):
    num = K.sum(true * pred)
    den = K.sqrt(K.sum(true * true)) * K.sqrt(K.sum(pred * pred))
    return -(num / (den +eps))


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'Weighted SDR Loss'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def mse_plus_weighted_SDR_loss(true_speech , pred_speech):

    noisy_speech=true_speech[:,:,1]
    true_speech=true_speech[:,:,0]
    pred_speech=pred_speech[:,:,0]
    
    
    def SDR_loss (pred, true, eps = 1e-8):
        num = K.sum(pred * true)
        den = K.sqrt(K.sum(true * true)) * K.sqrt(K.sum(pred * pred))
        return -(num / (den + eps))

    pred_noise = noisy_speech - pred_speech
    true_noise = noisy_speech - true_speech
    alpha      = K.sum(true_speech**2) / (K.sum(true_speech**2) + K.sum(true_noise**2)) 
    sound_SDR = SDR_loss(pred_speech, true_speech)
    noise_SDR = SDR_loss(pred_noise, true_noise)
    
    WSNR= alpha * sound_SDR + (1-alpha) * noise_SDR

    
    
    x_mag = tf.abs(
                tf.signal.stft(
                    signals=pred_speech,
                    frame_length=200,
                    frame_step=100,
                    fft_length=200,
                )
            )
    y_mag = tf.abs(
                tf.signal.stft(
                    signals=true_speech,
                    frame_length=200,
                    frame_step=100,
                    fft_length=200,
                )
            )


    mse = tf.keras.losses.MeanSquaredError()

    mse = tf.keras.losses.MeanSquaredError()
    mse=mse(y_mag, x_mag)
    
        
    return mse+10*WSNR
