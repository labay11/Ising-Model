import numpy as np
import matplotlib.pyplot as plt

def load_file(fiename):
	return np.loadtxt(fiename)

def plot(data):
	def set_limit_axis(axis, ys):
		miny, maxy = np.min(ys), np.max(ys)
		dminy, dmaxy = np.log10(np.abs(miny)), np.log10(np.abs(maxy))
		axis.set_ylim(miny - 10**(dmaxy-1), maxy + 10**(dmaxy-1))

	fig, axes = plt.subplots(2, 2, sharex='col')

	set_limit_axis(axes[0,0], data[:,1])
	axes[0,1].set_ylim(-1.1,1.1)
	set_limit_axis(axes[1,0], data[:,5])
	set_limit_axis(axes[1,1], data[:,6])

	#axes[0,0].set_xlabel(r'$T$')
	axes[0,0].set_ylabel(r'$E$')

	#axes[0,1].set_xlabel(r'$T$')
	axes[0,1].set_ylabel(r'$M$')

	axes[1,0].set_xlabel(r'$T$')
	axes[1,0].set_ylabel(r'$c_m$')

	axes[1,1].set_xlabel(r'$T$')
	axes[1,1].set_ylabel(r'$\chi$')

	axes[0,0].errorbar(data[:,0], data[:,1], yerr=data[:,2], marker='.', ls='None')
	axes[0,1].errorbar(data[:,0], data[:,3], yerr=data[:,4], marker='.', ls='None')
	axes[1,0].scatter(data[:,0], data[:,5], marker='o')	
	axes[1,1].scatter(data[:,0], data[:,6], marker='o')

	fig.tight_layout()

	plt.savefig('ising_tc.png')
	plt.show()

plot(load_file('ising_sim.txt'))