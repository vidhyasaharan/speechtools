# speechbox.py

import numpy as np
from scipy import fftpack as spfft


def specgram(x,fs,fsize_sec=0.02,fshift_sec=0.01):
    nsam = len(x)
    fsize = int(fsize_sec*fs)
    fshift = int(fshift_sec*fs)
    nframes = nsam//fshift
    ssize = int(np.ceil((fsize+1)/2))
    mspec = np.zeros((nframes,ssize))
    x = np.pad(x,(0,nframes*fshift+fsize-nsam),'constant')

    for i in range(nframes):
        mspec[i,:] = np.absolute(np.fft.rfft(x[i*fshift:i*fshift+fsize]))
    mspec = np.flipud((mspec.T))
    return mspec


def hz2mel(f_hz):
    return 1127*np.log(1 + (f_hz/700))


def mel2hz(f_mel):
    return 700*(np.exp(f_mel/1127)-1)


def melfbank(fs,nfft,numfilters):
    fmax = hz2mel(fs/2)
    fc = mel2hz(np.linspace(0,fmax,num=numfilters))
    filt = np.zeros((numfilters,nfft))
    fc_nfft_indx = np.around((nfft-1)*2*fc/fs)
    for i in range(numfilters-1):
        sdx = int(fc_nfft_indx[i])
        edx = int(fc_nfft_indx[i+1]) + 1    
        filt[i,sdx:edx] = np.linspace(1,0,edx-sdx)

    for i in range(1,numfilters):
        sdx = int(fc_nfft_indx[i-1])
        edx = int(fc_nfft_indx[i]) + 1
        filt[i,sdx:edx] = np.linspace(0,1,edx-sdx)
    return filt

    
def mfcc(x,fs,nmfcc=13,numfilters=17,fsize_sec=0.02,fshift_sec=0.01):
    nsam = len(x)
    fsize = int(fsize_sec*fs)
    fshift = int(fshift_sec*fs)
    win = np.hamming(fsize)    
    nframes = nsam//fshift
    ssize = int(np.ceil((fsize+1)/2))
    mfc = np.zeros((nframes,nmfcc))
    x = np.pad(x,(0,nframes*fshift+fsize-nsam),'constant')
    flt = melfbank(fs,ssize,numfilters)    
    
    for i in range(nframes):
        fmspec = np.absolute(np.fft.rfft(win*x[i*fshift:i*fshift+fsize]))
        fcep = spfft.dct(np.log(np.dot(flt,fmspec)),norm='ortho')
        mfc[i,:] = fcep[0:nmfcc]
    
    return mfc
