import matplotlib.pyplot as plt
import numpy as np


def fourier_total(freqw, data_dir, data_ortho,
                  xlim=(0, 30), ylim=(10e-15, 1),
                  xlabel=r'Frequency $\text{ in } \omega/\omega_0$',
                  ylabel=r'I_\mathrm{hh} $\text{ in atomic units}$',
                  paramlegend=None, supertitle=None, title=None, savename=None):
	"""
	Plots parallel and orthogonal data
	"""
	_fig, ax = plt.subplots(1)

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xticks(np.arange(xlim[1]+1))
	ax.grid(True, axis='x', ls='--')
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	data_total = data_dir + data_ortho
	for freq, data in zip(freqw, data_total):
		ax.semilogy(freq, data)

	if paramlegend is not None:
		ax.legend(paramlegend)

	if title is not None:
		plt.title(title)

	if supertitle is not None:
		plt.suptitle(supertitle)

	if savename is not None:
		plt.savefig(savename)
	else:
		plt.show()


def fourier_dir_ortho(freqw, data_dir, data_ortho, xlim=(0.2, 30),
                      ylim=(10e-15, 1),
                      xlabel=r'Frequency $\omega/\omega_0$',
                      ylabel=r'$I_\mathrm{hh}$ in atomic units',
                      paramlegend=None, ls_dir=None, ls_ortho=None,
                      supertitle=None, title=None, savename=None):

	freqw = freqw.real
	data_dir = data_dir.real
	data_ortho = data_ortho.real
	_fig, ax = plt.subplots(1)
	ax.set_xticks(np.arange(xlim[1]+1))
	ax.grid(True, axis='both', ls='--')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	lines_dir = ax.semilogy(freqw.T, data_dir.T)
	plt.gca().set_prop_cycle(None)
	lines_ortho = ax.semilogy(freqw.T, data_ortho.T, color='red', linestyle='--')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	if ls_dir:
		for ls, line in zip(ls_dir, lines_dir):
			line.set_linestyle(ls)
	if ls_ortho:
		for ls, line in zip(ls_ortho, lines_ortho):
			line.set_linestyle(ls)

	if paramlegend is not None:
		ax.legend(paramlegend)

	if title is not None:
		plt.title(title)

	if supertitle is not None:
		plt.suptitle(supertitle)

	if savename is None:
		plt.show()
	else:
		plt.savefig(savename)

def fourier_ana_num(freqw, data_1, data_2, xlim=(0.2, 30),
                    ylim=(10e-15, 1),
                    xlabel=r'Frequency $\omega/\omega_0$',
                    ylabel=r'$I_\mathrm{hh}$ in atomic units',
                    paramlegend=None, ls_dir=None, ls_ortho=None,
                    supertitle=None, title=None, savename=None):

	freqw = freqw.real
	data_dir = data_1.real
	data_ortho = data_2.real
	_fig, ax = plt.subplots(1)
	ax.set_xticks(np.arange(xlim[1]+1))
	ax.grid(True, axis='both', ls='--')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	lines_dir = ax.semilogy(freqw.T, data_1.T, label='analytical calculation')
	plt.gca().set_prop_cycle(None)
	lines_ortho = ax.semilogy(freqw.T, data_2.T, color='red', linestyle='--', label='numerical calculation')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.legend(loc='upper right')


	if ls_dir:
		for ls, line in zip(ls_dir, lines_dir):
			line.set_linestyle(ls)
	if ls_ortho:
		for ls, line in zip(ls_ortho, lines_ortho):
			line.set_linestyle(ls)

	if paramlegend is not None:
		ax.legend(paramlegend)

	if title is not None:
		plt.title(title)

	if supertitle is not None:
		plt.suptitle(supertitle)

	if savename is None:
		plt.show()
	else:
		plt.savefig(savename)



def fourier_dir_ortho_split(freqw, data_dir, data_ortho, xlim=(0.2, 30),
                            ylim=(10e-15, 1),
                            xlabel=r'Frequency $\omega/\omega_0$',
                            ylabel=r'$I_\mathrm{hh} \text{ in atomic units}$',
                            paramlegend=None, supertitle=None, savename=None):

	_fig, ax = plt.subplots(2)
	for a in ax:
		a.set_xlim(xlim)
		a.set_ylim(ylim)
		a.set_xticks(np.arange(31))
		a.grid(True, axis='x', ls='--')
		a.set_ylabel(ylabel)
	ax[0].set_title(r'$\mathbf{E}$ parallel')
	ax[1].set_title(r'$\mathbf{E}$ orthogonal')
	ax[1].set_xlabel(xlabel)
	for freq, data_d, data_o in zip(freqw, data_dir, data_ortho):
		ax[0].semilogy(freq, data_d)
		ax[1].semilogy(freq, data_o)

	if paramlegend is not None:
		ax[0].legend(paramlegend)

	if supertitle is not None:
		plt.suptitle(supertitle)

	if savename is None:
		plt.show()
	else:
		plt.savefig(savename)


def fourier_dir_ortho_angle(freqw, re_j_E_dir, im_j_E_dir, 
                            re_j_ortho, im_j_ortho,
                            xlim=(0.2, 30), ylim=(-90, 90),
                            xlabel=r'Frequency $f/f_0$',
                            ylabel=r'$\alpha$',
                            ls='-', marker='.',
                            paramlegend=None, supertitle=None, title=None, savename=None):

	_fig, ax = plt.subplots(1)
	freqw = freqw.real
	ax.set_xticks(np.arange(xlim[1]+1))
	ax.grid(True, axis='x', ls='--')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	anglearr = np.linspace(-np.pi/2, np.pi/2, 181)
	cosarr = np.cos(anglearr)
	sinarr = np.sin(anglearr)

	J_E_dir = re_j_E_dir + 1j * im_j_E_dir
	J_ortho = re_j_ortho + 1j * im_j_ortho
	alphaarr = np.empty(J_E_dir.size, dtype=np.float64)
	for i, (jd, jo) in enumerate(zip(J_E_dir, J_ortho)):
		maxidx = np.argmax(np.abs(cosarr * jd + sinarr * jo))
		alphaarr[i] = anglearr[maxidx]

	_angle = ax.plot(freqw, np.rad2deg(alphaarr), ls=ls, marker=marker)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	if paramlegend is not None:
		ax.legend(paramlegend)

	if title is not None:
		plt.title(title)

	if supertitle is not None:
		plt.suptitle(supertitle)

	if savename is None:
		plt.show()
	else:
		plt.savefig(savename)

def fourier_dir_ortho_angle_polar(freqw, re_j_E_dir, im_j_E_dir, re_j_ortho, im_j_ortho,
                                  savename=None):
	fig = plt.figure()
	fig.suptitle(r'HH in $\omega/\omega_0$')

	J_E_dir = re_j_E_dir + 1j*im_j_E_dir
	J_ortho = re_j_ortho + 1j*im_j_ortho

	N = 10
	angle_arr = np.linspace(-np.pi/2, np.pi/2, 181)
	angle_plot_arr = np.linspace(0, np.pi, 181)

	cos_arr = np.cos(angle_arr)
	sin_arr = np.sin(angle_arr)
	offset = 4190

	for i in range(N):
		ax = fig.add_subplot(1, N, i+1, projection='polar')
		# ax.set_yticklabels([""])

		pw = np.abs(cos_arr * J_E_dir[offset + i] + sin_arr * J_ortho[offset + i])
		ax.plot(angle_plot_arr, pw)
		rmax = ax.get_rmax()
		ax.set_rmax(1.1*rmax)
		freqstring = '{:.2f}'.format(freqw[offset + i])
		ax.set_title(r'$' + freqstring + r'$', va='top', pad=30)
		if i == 0:
			ax.set_rgrids(np.linspace(0.25, 1, 4)*rmax, labels=None, angle=None, fmt=None)
			xticks = np.linspace(0, np.pi, 7)
			ax.set_xlim([0, np.pi])
			ax.set_xticks(xticks)
			ax.set_xticklabels(['{:.0f}'.format(deg) for deg in np.rad2deg(xticks - np.pi/2)])
		else:
			ax.set_rgrids(np.linspace(0.25, 1, 4)*rmax, labels=None, angle=None, fmt=None)
			ax.set_xlim([0, np.pi])
			ax.set_xticks(np.linspace(0, np.pi, 7))
			# ax.set_xticklabels(["" for i in range(7)])

	if savename is None:
		plt.show()
	else:
		plt.savefig(savename)
