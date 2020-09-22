import matplotlib.pyplot as plt
import h5py
import numpy as np

def read_master_results_hdf5(filename):
    print('Reading file: {}'.format(filename))
    hf = h5py.File(filename,'r')
    print()
    print('This file has the following keys')
    print(hf.keys())
    print()
    print(hf['general/description'].value)
    return hf


def get_cmap_colors(cmap='jet',p=None,N=10):
    cm = plt.get_cmap(cmap)
    if p is None:
        return [cm(i) for i in np.linspace(0,1,N)]
    else:
        normalize = matplotlib.colors.Normalize(vmin=min(p), vmax=max(p))
        colors = [cm(normalize(value)) for value in p]
        return colors


### Reading in hdf5 file ###
hf = read_master_results_hdf5('GJ_699_results.hdf5')
hf

hf.keys()

hf['template'].keys()

hf['general'].keys()

hf['rv'].keys()

### Plotting CCFs ###
hf['rv/chi2map'].value.shape


# original # 
def plot_chi2maps(vgrid,chi2map,orders=[4,5,6,14,15,16,17,18],savename='',targetname='',cmap="coolwarm"):

    '''
    Function to plot chi2maps
    EXAMPLE:
    hf = read_master_results_hdf5('GJ_4037_results.hdf5')
    v = hf['rv/vgrid'].value
    chi2map = hf['rv/chi2map'].value
    plot_chi2maps(v,chi2map)
    '''
    fig, axx = plt.subplots(nrows=2,ncols=4,figsize=(12,6),sharex=True)

    for i,o in enumerate(orders):
        xx = axx.flatten()[i]
        colors = get_cmap_colors(N=chi2map.shape[0],cmap=cmap)
        for j in range(chi2map.shape[0]):
        # for j in range(100):
            xx.plot(vgrid,chi2map[j][o]/np.median(chi2map[j][o]),color=colors[j])
            xx.set_title('o={}'.format(o))
            xx.set_xlabel('V [km/s]',labelpad=1)
            if i%4==0:
                xx.set_ylabel('$\chi^2$')
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle(targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=200)
        print('Saved to {}'.format(savename))

plot_chi2maps(v,chi2map, savename='output-0.png', targetname='GJ 699')
plt.close()        


# modified-1 # 
def plot_chi2maps(vgrid,chi2map,orders=[4,5,6,14,15,16,17,18],savename='',targetname='',cmap="coolwarm"):

    fig, axx = plt.subplots(nrows=2,ncols=4,figsize=(12,6),sharex=True)

    for i,o in enumerate(orders):
        xx = axx.flatten()[i]
        colors = get_cmap_colors(N=chi2map.shape[0],cmap=cmap)
        for j in range(chi2map.shape[0]):
            xx.plot(vgrid,chi2map[j][o],color=colors[j], alpha = 0.1)
            xx.set_title('o={}'.format(o))
            xx.set_xlabel('V [km/s]',labelpad=1)
            xx.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if i%4==0:
                xx.set_ylabel('$\chi^2$')
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle(targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=200)
        print('Saved to {}'.format(savename))

v = hf['rv/vgrid'].value
idx = (abs(v)<2.5)
chi2map = hf['rv/chi2map'].value
plot_chi2maps(v,chi2map, savename='output-1.png', targetname='GJ 699')
plt.close()



colors = get_cmap_colors(N=chi2map.shape[0],cmap="coolwarm")
sum_chi2map = np.nansum(chi2map, axis=1)
for j in range(chi2map.shape[0]):
    plt.plot(v,sum_chi2map[j], color=colors[j], alpha = 0.1)
plt.xlabel('V [km/s]',labelpad=1)
plt.ylabel('$\chi^2$')
plt.savefig('chi2.png')
plt.show()


for xx in np.arange(100)/100+1:

    y = np.exp(-sum_chi2map[j]/np.median(sum_chi2map[j])/xx)
    v_range = v[y > max(y)/2]
    FWHM = max(v_range) - min(v_range)
    if FWHM>5.451:
        print(xx)


colors = get_cmap_colors(N=chi2map.shape[0],cmap="coolwarm")
sum_chi2map = np.nansum(chi2map, axis=1)
for j in range(chi2map.shape[0]):
    plt.plot(v,np.exp(-sum_chi2map[j]/np.median(sum_chi2map[j])/1.43), color=colors[j], alpha = 0.1)
plt.xlabel('V [km/s]',labelpad=1)
plt.ylabel('$f(\chi^2)$')
plt.savefig('f(chi2).png')
plt.show()






    fig, axx = plt.subplots(nrows=2,ncols=4,figsize=(12,6),sharex=True)

    for i,o in enumerate(orders):
        xx = axx.flatten()[i]
        colors = get_cmap_colors(N=chi2map.shape[0],cmap=cmap)
        for j in range(chi2map.shape[0]):
            xx.plot(vgrid,chi2map[j][o],color=colors[j], alpha = 0.1)
            xx.set_title('o={}'.format(o))
            xx.set_xlabel('V [km/s]',labelpad=1)
            xx.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if i%4==0:
                xx.set_ylabel('$\chi^2$')
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle(targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=200)
        print('Saved to {}'.format(savename))

v = hf['rv/vgrid'].value
chi2map = hf['rv/chi2map'].value
plot_chi2maps(v,chi2map, savename='output-1.png', targetname='GJ 699')
plt.close()




















# modified-2 #
def plot_chi2maps(vgrid,chi2map,orders=[4,5,6,14,15,16,17,18],savename='',targetname='',cmap="coolwarm"):

    fig, axx = plt.subplots(nrows=2,ncols=4,figsize=(12,6),sharex=True)

    for i,o in enumerate(orders):
        xx = axx.flatten()[i]
        colors = get_cmap_colors(N=chi2map.shape[0],cmap=cmap)
        for j in range(10):
            xx.plot(vgrid, np.exp(-chi2map[j][o]/np.min(chi2map[j][o])/10),color=colors[j],alpha = 0.1)
            xx.set_title('o={}'.format(o))
            xx.set_xlabel('V [km/s]',labelpad=1)
            xx.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            if i%4==0:
                xx.set_ylabel('$f(\chi^2)$')
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle(targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=200)
        print('Saved to {}'.format(savename))

plot_chi2maps(v,chi2map, savename='output-2.png', targetname='GJ 699')
plt.close()


plt.plot(v, chi2map[0,0,:])
plt.show()









