#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os

class PreProcess(object):
    def __init__(self, *args, **kwargs):
        super(PreProcess, self).__init__(*args, **kwargs)
        self.dir = '/data/dell5/userdir/maotx/DSC/data'

    def _convert(self, fname='small.npy'):
        data = np.load('/data/dell5/userdir/jyj/LAMOST/bstars_info.npy')
        selnum = np.unique(np.random.randint(0,len(data),2000))[:1000]
        data = data[selnum]
        np.save(os.path.join(self.dir,fname),data)
        
    def GaussSampling(self, x, flux, wl, sigma, Iv=None):
        '''
        Iv: inverse variance weight
        wl: wave length
        x: wanted wave length
        '''
        def GaussKernel(x, mu=0., sigma=0.):
            # dwl = 1.38, so sigma should greater than 1.38 at least
            return 1./np.sqrt(2*np.pi*sigma**2.)*np.exp((x-mu)**2./(2*sigma**2.))

        if x[0] < 3700.+3.*sigma or x[-1] > 9100.-3.*sigma:
            print 'selected wavelength should from %d to %d...' %(3700.+3.*sigma,9100.-3.*sigma)
            exit()

        flux_new = []
        for i in xrange(len(x)):
            bool = (wl<(x[i]+3*sigma))*(wl>(x[i]-3*sigma))
            flux_i = flux[bool]
            wl_i = wl[bool]
            w = GaussKernel(wl_i,x[i],sigma)
            if Iv is not None:
                Iv_i = Iv[bool]
                value = np.sum(flux_i*w * Iv_i)/np.sum(w*Iv_i)
            else:
                value = np.sum(flux_i*w)/np.sum(w)
            flux_new.append(value)
        flux_new = np.array(flux_new)
        return flux_new
        
    def run_GS(self, data, sigma=3., IsIv=True):
        wl_new = np.linspace(4000.,8800.,4000)
        flux = data['flux']
        wl = data['wavelen']
        if IsIv:
            Iv = data['snr']
            Iv *= (1-data['snr_mask_2'])
            flux_new = [self.GaussSampling(wl_new, flux[i], wl[i], sigma, Iv=Iv[i]) for i in xrange(data.shape[0])]
        else:
            flux_new = [self.GaussSampling(wl_new, flux[i], wl[i], sigma, Iv=None) for i in xrange(data.shape[0])]
        return flux_new


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = PreProcess()

    data = np.load('./data/small.npy')
    flux = data['flux']
    wl = data['wavelen']
    #----------------------------------------
    # get inverse variance
    data['snr'][wl!=0]/=flux[wl!=0]
    data['snr']=data['snr']**2.
    data['snr'][np.isnan(data['snr'])]=0. # get inverse variance, and set it zero at flux is 0
    #----------------------------------------
#   s = data[4]
#   print s['obsid']
#   print s['flux'][-200:]
#   print s['wavelen'][-200:]
#   print s['snr'][-200:]
#   plt.plot(s['wavelen'], s['flux'])
#   plt.show()
#   exit()
    flnew = test.run_GS(data, sigma=3., IsIv=True)
    s = np.vstack(flnew)
    bool = (np.isnan(s).sum(axis=1)==0)
    data = data[bool]
    data['flux'] = s[bool]
    np.save('./data/small_flux_resample.npy', data)
