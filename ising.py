import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

k_B = 1 #1.38064852e-23

levels = [-2, 0, 2]
colors = ['b', 'r']
cmap_spins, norm_spins = clrs.from_levels_and_colors(levels, colors)

class Ising(animation.TimedAnimation):
	"""docstring for Ising"""
	def __init__(self, lattice_size, H, J, T, spin=0.5, iters=100000, freq_measure=100):
		self.size = lattice_size
		self.N = self.size[0] * self.size[1]

		# number of possible spin states
		self.dj = int(2 * spin) + 1 

		self.T = T
		self.beta = 1 / (k_B * T)
		self.H = H
		self.J = J

		self.iters = iters
		self.freq_measure = freq_measure
		self.s_flips = 10 #int(np.sqrt(self.N))

		self.init_spins()
		self.init_figure()

		animation.TimedAnimation.__init__(self, self.fig, interval=1, blit=True, repeat=False)

	def init_figure(self):
		self.fig = plt.figure()

		self.ax_spins = plt.subplot2grid((2,2), (0, 0)) # spins
		self.ax_eng = plt.subplot2grid((2,2), (0, 1)) # magnetization
		self.ax_mag = plt.subplot2grid((2,2), (1, 0)) # energy

		self.ax_eng.set_xlabel(r'$t$')
		self.ax_eng.set_ylabel(r'$E$')

		self.ax_mag.set_xlabel(r'$t$')
		self.ax_mag.set_ylabel(r'$M$')

		self.line_spins = self.ax_spins.imshow(self.A, cmap='Blues', norm = clrs.Normalize(vmin=-self.dj / 2.0, vmax=self.dj / 2.0)) #cmap = cmap_spins, norm = norm_spins)
		self.line_eng = self.ax_eng.plot([],[],'b')[0]
		self.line_mag = self.ax_mag.plot([],[],'r')[0]

		self.ax_eng.set_xlim(0, self.iters)
		self.ax_mag.set_xlim(0, self.iters)

		self.fig.colorbar(self.line_spins, extend='both', shrink=0.9, ax=self.ax_spins)

		self._drawn_artists = [self.line_spins, self.line_eng, self.line_mag]

	def init_spins(self):
		# create a random 2D array of spins
		self.A = np.random.randint(0, self.dj, size=self.size, dtype=int) * 2 - 1

		self.times = np.linspace(0, self.iters, self.iters+1)
		self.samples = np.linspace(0, self.iters, self.iters // self.freq_measure + 1)

		self.data = np.zeros((self.iters // self.freq_measure + 1, 2)) # [energy, mag]

	def set_temperature(self, t):
		self.T = t
		self.beta = 1 / (k_B * t)

	def total_energy(self):
		nn = np.roll(self.A, 1, 0) + np.roll(self.A, -1, 0) + np.roll(self.A, 1, 1) + np.roll(self.A, -1, 1)
		return - np.sum((self.H + self.J * nn) * self.A) / (4 * self.N)

	def total_magnetisation(self):
		return np.sum(self.A) / self.N

	def energy_spin(self, i, j):
		s_neigh = self.A[(i+1) % self.size[0],j] + self.A[(i-1) % self.size[0],j] + \
					self.A[i, (j+1) % self.size[1]] + self.A[i,(j-1) % self.size[1]]

		return - (self.H + self.J * s_neigh) * self.A[i,j] / self.N

	def get_avg_energy(self, from_it = None, power = 1):
		if from_it is None:
			from_it = self.iters // (2 * self.freq_measure)

		return np.mean(self.data[from_it:,0]**power), np.std(self.data[from_it:,0]**power)

	def get_avg_magnetisation(self, from_it = None, power = 1):
		if from_it is None:
			from_it = self.iters // (2 * self.freq_measure)

		return np.mean(self.data[from_it:,1]**power), np.std(self.data[from_it:,1]**power)

	def start_mc(self):
		self.init_spins() # reset spins

		print("Starting Simulation (it = %d, T = %.2f)..." % (self.iters, self.T), end=' ')
		for k in range(self.iters):
			self.run_mc(k)

		print("Ok!")

		return True

	def run_mc(self, k):
		# choose the spins randomly
		mc_randx = np.random.randint(0, self.size[0], size = self.s_flips)
		mc_randy = np.random.randint(0, self.size[1], size = self.s_flips)

		coins = np.random.rand(self.s_flips)

		nn_sum = np.roll(self.A, 1, 0) + np.roll(self.A, -1, 0) + np.roll(self.A, 1, 1) + np.roll(self.A, -1, 1)
		dE = 2 * (self.H + self.J * nn_sum[mc_randx, mc_randy]) * self.A[mc_randx,mc_randy] 
		self.A[mc_randx,mc_randy] *= np.where(coins < np.exp(- self.beta * dE), -1, 1)

		if k % self.freq_measure == 0:
			self.data[k//self.freq_measure,0] = self.total_energy()
			self.data[k//self.freq_measure,1] = self.total_magnetisation()

	def _draw_frame(self, k):
		k = int(k)
		sample = k // self.freq_measure

		self.run_mc(k)

		self.line_spins.set_data(self.A)
		self.line_eng.set_data(self.samples[:sample+1], self.data[:sample+1,0])
		self.line_mag.set_data(self.samples[:sample+1], self.data[:sample+1,1])

		self.ax_eng.set_ylim(np.min(self.data[:sample+1,0]), np.max(self.data[:sample+1,0]))
		self.ax_mag.set_ylim(np.min(self.data[:sample+1,1]), np.max(self.data[:sample+1,1]))

	def new_frame_seq(self):
		return iter(self.times)

lattice = (30, 30)		# lattice size
H = 0					# externa field [-1,1]
J = 0.5					# coupling between spins [-1, 1]
T = 1					# temperature
iters = 10000			# number of iteration (~10000 to stabilize with 30x30 lattice)

ani = Ising(lattice, H, J, T, iters=iters, spin=0.5, freq_measure=10)
#ani.start_mc()  # run simulation in background
plt.show()		 # run simulation while showing the plots

E, uE = ani.get_avg_energy()
E2, uE2 = ani.get_avg_energy(power=2)
M, uM = ani.get_avg_magnetisation()
M2, uM2 = ani.get_avg_magnetisation(power=2)

print('Energy: %.2g +/- %.2g' % (E, uE))
print('Magnetisation: %.2g +/- %.2g' % (M, uM))