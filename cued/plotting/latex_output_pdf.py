import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
import tikzplotlib

from cued.plotting.colormap import whitedarkjet
from cued.plotting import contourf_remove_white_lines, label_inner, init_matplotlib_config, unit, symb
from cued.utility import ConversionFactors as CoFa, rmdir_mkdir_chdir, cued_copy, chdir
from cued.kpoint_mesh import hex_mesh, rect_mesh

init_matplotlib_config()

def conditional_pdflatex(name_data, name_tex):
	"""
	Check wheter pdflatex compiler exists before proceeding.
	"""
	print("======================")
	if shutil.which("pdflatex"):
		print("Creating PDF")
		print(name_data)
		os.system("pdflatex " + name_tex + " > /dev/null 2>&1")
		os.system("pdflatex " + name_tex + " > /dev/null 2>&1")
		print("Done")
	else:
		print("No LaTeX (pdflatex) compiler found, only keeping .tex files and logo.")
		print("Can be compiled by hand by using 'pdflatex' on the .tex file.")
	print("======================")


def write_and_compile_latex_PDF(T, W, P, sys, Mpi):

	t_fs = T.t*CoFa.au_to_fs
	num_points_for_plotting = 960

	t_idx = get_time_indices_for_plotting(T.E_field, t_fs, num_points_for_plotting)
	f_idx = get_freq_indices_for_plotting(W.freq/P.f, num_points_for_plotting, freq_max=30)

	t_idx_whole = get_indices_for_plotting_whole(t_fs, num_points_for_plotting, start=0)
	f_idx_whole = get_indices_for_plotting_whole(W.freq, num_points_for_plotting, start=f_idx[0])

	high_symmetry_path_BZ = get_symmetry_path_in_BZ(P, num_points_for_plotting)

	latex_dir = P.header + "latex_pdf_files"

	rmdir_mkdir_chdir(latex_dir)

	cued_copy('plotting/tex_templates/CUED_summary.tex', '.')
	cued_copy('plotting/tex_templates/CUED_aliases.tex', '.')
	cued_copy('branding/logo.pdf', '.')

	write_parameters(P, Mpi)

	tikz_time(T.E_field*CoFa.au_to_MVpcm, t_fs, t_idx, r'E-field ' + unit['E'], "Efield")
	tikz_time(T.A_field*CoFa.au_to_MVpcm*CoFa.au_to_fs, t_fs, t_idx, r'A-field ' + unit['A'], "Afield")

	K = BZ_plot(P, T.A_field)

	bandstruc_and_dipole_plot_high_symm_line(high_symmetry_path_BZ, P, num_points_for_plotting, sys)

	dipole_quiver_plots(K, P, sys)

	density_matrix_plot(P, T, K)

	tikz_time(T.j_E_dir, t_fs, t_idx,
	          r'Current $j_{\parallel}(t)$ parallel to $\bE$ in atomic units', "j_E_dir")
	tikz_time(T.j_E_dir, t_fs, t_idx_whole,
	          r'Current $j_{\parallel}(t)$ parallel to $\bE$ in atomic units', "j_E_dir_whole_time")
	tikz_time(T.j_ortho, t_fs, t_idx,
	          r'Current $j_{\bot}(t)$ orthogonal to $\bE$ in atomic units', "j_ortho")
	tikz_time(T.j_ortho, t_fs, t_idx_whole,
	          r'Current $j_{\bot}(t)$ orthogonal to $\bE$ in atomic units', "j_ortho_whole_time")
	tikz_freq(W.I_E_dir, W.I_ortho, W.freq/P.f, f_idx_whole,
	          r'Emission intensity in atomic units', "Emission_para_ortho_full_range", two_func=True,
	          label_1="$\;I_{\parallel}(\w)$", label_2="$\;I_{\\bot}(\w)$")
	tikz_freq(W.I_E_dir, W.I_ortho, W.freq/P.f, f_idx,
	          r'Emission intensity in atomic units', "Emission_para_ortho", two_func=True,
	          label_1="$\;I_{\parallel}(\w)$", label_2="$\;I_{\\bot}(\w)$")
	tikz_freq(W.I_E_dir + W.I_ortho, None, W.freq/P.f, f_idx,
	          r'Emission intensity in atomic units', "Emission_total", two_func=False,
	          label_1="$\;I(\w) = I_{\parallel}(\w) + I_{\\bot}(\w)$")
	tikz_freq(W.I_E_dir_hann + W.I_ortho_hann, W.I_E_dir_parzen+W.I_ortho_parzen, W.freq/P.f, f_idx,
	          r'Emission intensity in atomic units', "Emission_total_hann_parzen", two_func=True,
	          label_1="$\;I(\w)$ with $\\bj(\w)$ computed using the Hann window",
	          label_2="$\;I(\w)$ with $\\bj(\w)$ computed using the Parzen window", dashed=True)

	replace("semithick", "thick", "*")

	conditional_pdflatex(P.header.replace('_', ' '), 'CUED_summary.tex')
	chdir()


def write_parameters(P, Mpi):

	if P.BZ_type == 'rectangle':
		if P.angle_inc_E_field == 0:
			replace("PH-EFIELD-DIRECTION", "$\\\\hat{e}_\\\\phi = \\\\hat{e}_x$")
		elif P.angle_inc_E_field == 90:
			replace("PH-EFIELD-DIRECTION", "$\\\\hat{e}_\\\\phi = \\\\hat{e}_y$")
		else:
			replace("PH-EFIELD-DIRECTION", "$\\\\phi = "+str(P.angle_inc_E_field)+"^\\\\circ$")
	elif P.BZ_type == 'hexagon':
		if P.align == 'K':
			replace("PH-EFIELD-DIRECTION", "$\\\\Gamma$-K direction")
		elif P.align == 'M':
			replace("PH-EFIELD-DIRECTION", "$\\\\Gamma$-M direction")

	if P.user_defined_field:
		replace("iftrue", "iffalse")
	else:
		replace("PH-E0",	str(P.E0_MVpcm))
		replace("PH-FREQ",	str(P.f_THz))
		replace("PH-CHIRP", str(P.chirp_THz))
		eps = 1.0E-13
		if P.phase > np.pi/2-eps and P.phase < np.pi/2+eps:
			 replace("PH-CEP", "\\\\pi\/2")
		elif P.phase > np.pi-eps and P.phase < np.pi+eps:
			 replace("PH-CEP", "\\\\pi")
		elif P.phase > 3*np.pi/2-eps and P.phase < 3*np.pi/2+eps:
			 replace("PH-CEP", "3\\\\pi\/2")
		elif P.phase > 2*np.pi-eps and P.phase < 2*np.pi+eps:
			 replace("PH-CEP", "2\\\\pi")
		else:
			 replace("PH-CEP", str(P.phase))
		replace("PH-SIGMA", str(P.sigma_fs))
		replace("PH-FWHM", '{:.3f}'.format(P.sigma_fs*2*np.sqrt(np.log(2))))

	replace("PH-BZ", P.BZ_type)
	replace("PH-NK1", str(P.Nk1))
	replace("PH-NK2", str(P.Nk2))
	replace("PH-T2", str(P.T2_fs))
	replace("PH-RUN", '{:.1f}'.format(P.run_time))
	replace("PH-MPIRANKS", str(Mpi.size))


def tikz_time(func_of_t, time_fs, t_idx, ylabel, filename):

	_fig, (ax1) = plt.subplots(1)
	_lines_exact_E_dir = ax1.plot(time_fs[t_idx], func_of_t[t_idx], marker='')

	t_lims = (time_fs[t_idx[0]], time_fs[t_idx[-1]])

	ax1.grid(True, axis='both', ls='--')
	ax1.set_xlim(t_lims)
	ax1.set_xlabel(unit['t'])
	ax1.set_ylabel(ylabel)
	# ax1.legend(loc='upper right')

	tikzplotlib.save(filename + ".tikz", axis_height=r'\figureheight',
	                 axis_width=r'\figurewidth' )
	# Need to explicitly close figure after writing
	plt.close(_fig)


def tikz_freq(func_1, func_2, freq_div_f0, f_idx, ylabel, filename, two_func, \
              label_1=None, label_2=None, dashed=False):

	xlabel = r'Harmonic order = ' + unit['ff0']

	_fig, (ax1) = plt.subplots(1)
	_lines_exact_E_dir = ax1.semilogy(freq_div_f0[f_idx], func_1[f_idx], marker='', label=label_1)
	if two_func:
		if dashed:
			_lines_exact_E_dir = ax1.semilogy(freq_div_f0[f_idx], func_2[f_idx], marker='', label=label_2, \
			                                  linestyle='--')
		else:
			_lines_exact_E_dir = ax1.semilogy(freq_div_f0[f_idx], func_2[f_idx], marker='', label=label_2)


	f_lims = (freq_div_f0[f_idx[0]], freq_div_f0[f_idx[-1]])

	ax1.grid(True, axis='both', ls='--')
	ax1.set_xlim(f_lims)
	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel)
	ax1.legend(loc='upper right')
	ax1.set_xticks(np.arange(f_lims[1]+1))

	tikzplotlib.save(filename + ".tikz",
					 axis_height='\\figureheight',
					 axis_width ='\\figurewidth' )

	replace("xmax=30,", "xmax=30, xtick={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"+\
	        ",21,22,23,24,25,26,27,28,29,30}, xticklabels={,1,,,,5,,,,,10,,,,,15,,,,,20,,,,,25,,,,,30},", \
	        filename=filename + ".tikz")
	# Need to explicitly close figure after writing
	plt.close(_fig)


def replace(old, new, filename="CUED_summary.tex"):

	os.system("sed -i -e \'s/"+old+"/"+new+"/g\' "+filename)


def get_time_indices_for_plotting(E_field, time_fs, num_t_points_max):

	E_max = np.amax(np.abs(E_field))

	threshold = 1.0E-3

	index_t_plot_start = np.argmax(np.abs(E_field) > threshold*E_max)
	index_t_plot_end = E_field.size + 1 - np.argmax(np.abs(E_field[::-1]) > threshold*E_max)

	if index_t_plot_end - index_t_plot_start < num_t_points_max:
		step = 1
	else:
		step = (index_t_plot_end - index_t_plot_start)//num_t_points_max

	t_idx = range(index_t_plot_start, index_t_plot_end, step)

	return t_idx


def get_indices_for_plotting_whole(data, num_points_for_plotting, start):

	n_data_points = data.size
	step = n_data_points//num_points_for_plotting

	idx = range(start, n_data_points-1, step)

	return idx


def get_freq_indices_for_plotting(freq_div_f0, num_points_for_plotting, freq_min=-1.0E-8, freq_max=30):

	index_f_plot_start = np.argmax(freq_div_f0 > freq_min)
	index_f_plot_end = np.argmax(freq_div_f0 > freq_max)

	if index_f_plot_end - index_f_plot_start < num_points_for_plotting:
		step = 1
	else:
		step = (index_f_plot_end - index_f_plot_start)//num_points_for_plotting

	f_idx = range(index_f_plot_start, index_f_plot_end, step)

	return f_idx

def get_symmetry_path_in_BZ(P, num_points_for_plotting):

	Nk_per_line = num_points_for_plotting//2

	delta = 1/(2*Nk_per_line)
	neg_array_direct  = np.linspace(-1.0+delta, 0.0-delta, num=Nk_per_line)
	neg_array_reverse = np.linspace( 0.0+delta, -1.0-delta, num=Nk_per_line)
	pos_array_direct  = np.linspace( 0.0+delta, 1.0-delta, num=Nk_per_line)
	pos_array_reverse = np.linspace( 1.0-delta, 0.0+delta, num=Nk_per_line)

	if P.BZ_type == 'hexagon':

		R = 4.0*np.pi/(3*P.a)
		r = 2.0*np.pi/(np.sqrt(3)*P.a)
		vec_M = np.array( [ r*np.cos(-np.pi/6), r*np.sin(-np.pi/6) ] )
		vec_K = np.array( [ R, 0] )

		path = []
		for alpha in pos_array_reverse:
			kpoint = alpha*vec_K
			path.append(kpoint)

		for alpha in pos_array_direct:
			kpoint = alpha*vec_M
			path.append(kpoint)

	elif P.BZ_type == 'rectangle':

		vec_k_E_dir = P.length_BZ_E_dir*P.E_dir
		vec_k_ortho = P.length_BZ_ortho*np.array([P.E_dir[1], -P.E_dir[0]])

		path = []
		for alpha in pos_array_reverse:
			kpoint = alpha*vec_k_E_dir
			path.append(kpoint)

		for alpha in pos_array_direct:
			kpoint = alpha*vec_k_ortho
			path.append(kpoint)

	return np.array(path)


def BZ_plot(P, A_field):
	"""
		Function that plots the Brillouin zone
	"""
	BZ_fig = plt.figure(figsize=(10, 10))
	plt.plot(np.array([0.0]), np.array([0.0]), color='black', marker="o", linestyle='None')
	plt.text(0.01, 0.01, symb['G'])
	default_width = 0.87

	if P.BZ_type == 'hexagon':
		R = 4.0*np.pi/(3*P.a_angs)
		r = 2.0*np.pi/(np.sqrt(3)*P.a_angs)
		plt.plot(np.array([r*np.cos(-np.pi/6)]), np.array([r*np.sin(-np.pi/6)]), color='black', marker="o", linestyle='None')
		plt.text(r*np.cos(-np.pi/6)+0.01, r*np.sin(-np.pi/6)-0.05, symb['M'])
		plt.plot(np.array([R]), np.array([0.0]), color='black', marker="o", linestyle='None')
		plt.text(R, 0.02, r'K')
		kx_BZ = R*np.array([1,2,1,0.5,1,	0.5,-0.5,-1,   -0.5,-1,-2,-1,-0.5,-1,	 -0.5,0.5, 1,	  0.5,	1])
		tmp = np.sqrt(3)/2
		ky_BZ = R*np.array([0,0,0,tmp,2*tmp,tmp,tmp, 2*tmp,tmp, 0, 0, 0, -tmp,-2*tmp,-tmp,-tmp,-2*tmp,-tmp,0])
		plt.plot(kx_BZ, ky_BZ, color='black' )
		length		   = 5.0/P.a_angs
		length_x	   = length
		length_y	   = length
		ratio_yx	   = default_width
		dist_to_border = 0.1*length

	elif P.BZ_type == 'rectangle':
		# polar angle of upper right point of a rectangle that is horizontally aligned
		alpha = np.arctan(P.length_BZ_ortho/P.length_BZ_E_dir)
		beta  = P.angle_inc_E_field/360*2*np.pi
		dist_edge_to_Gamma = np.sqrt(P.length_BZ_E_dir**2+P.length_BZ_ortho**2)/2/CoFa.au_to_as
		kx_BZ = dist_edge_to_Gamma*np.array([np.cos(alpha+beta),np.cos(np.pi-alpha+beta),np.cos(alpha+beta+np.pi),np.cos(2*np.pi-alpha+beta),np.cos(alpha+beta)])
		ky_BZ = dist_edge_to_Gamma*np.array([np.sin(alpha+beta),np.sin(np.pi-alpha+beta),np.sin(alpha+beta+np.pi),np.sin(2*np.pi-alpha+beta),np.sin(alpha+beta)])
		plt.plot(kx_BZ, ky_BZ, color='black' )
		X_x = (kx_BZ[0]+kx_BZ[3])/2
		X_y = (ky_BZ[0]+ky_BZ[3])/2
		Y_x = (kx_BZ[0]+kx_BZ[1])/2
		Y_y = (ky_BZ[0]+ky_BZ[1])/2
		plt.plot(np.array([X_x,Y_x]), np.array([X_y,Y_y]), color='black', marker="o", linestyle='None')
		plt.text(X_x, X_y, symb['X'])
		plt.text(Y_x, Y_y, symb['Y'])

		dist_to_border = 0.1*max(np.amax(kx_BZ), np.amax(ky_BZ))
		length_x = np.amax(kx_BZ) + dist_to_border
		length_y = np.amax(ky_BZ) + dist_to_border
		ratio_yx = length_y/length_x*default_width
	if P.gauge == "velocity":
		Nk1_max = 120
		Nk2_max = 30
	else:
		Nk1_max = 24
		Nk2_max = 6

	if P.Nk1 <= Nk1_max and P.Nk2 <= Nk2_max:
		printed_paths = P.paths
		Nk1_plot = P.Nk1
		Nk2_plot = P.Nk2
	else:
		Nk1_safe = P.Nk1
		Nk2_safe = P.Nk2
		P.Nk1 = min(P.Nk1, Nk1_max)
		P.Nk2 = min(P.Nk2, Nk2_max)
		Nk1_plot = P.Nk1
		Nk2_plot = P.Nk2
		P.Nk = P.Nk1*P.Nk2
		if P.BZ_type == 'hexagon':
			dk, kweight, printed_paths, printed_mesh = hex_mesh(P)
		elif P.BZ_type == 'rectangle':
			dk, kweight, printed_paths, printed_mesh = rect_mesh(P)
		P.Nk1 = Nk1_safe
		P.Nk2 = Nk2_safe
		P.Nk = P.Nk1*P.Nk2

	plt.xlim(-length_x, length_x)
	plt.ylim(-length_y, length_y)

	plt.xlabel(unit['kx'])
	plt.ylabel(unit['ky'])

	for path in printed_paths:
		num_k				 = np.size(path[:,0])
		plot_path_x			 = np.zeros(num_k+1)
		plot_path_y			 = np.zeros(num_k+1)
		plot_path_x[0:num_k] = 1/CoFa.au_to_as*path[0:num_k, 0]
		plot_path_x[num_k]	 = 1/CoFa.au_to_as*path[0, 0]
		plot_path_y[0:num_k] = 1/CoFa.au_to_as*path[0:num_k, 1]
		plot_path_y[num_k]	 = 1/CoFa.au_to_as*path[0, 1]

		if P.gauge == "length":
			plt.plot(plot_path_x, plot_path_y)
		plt.plot(plot_path_x, plot_path_y, color='gray', marker="o", linestyle='None')

	A_min = np.amin(A_field)/CoFa.au_to_as
	A_max = np.amax(A_field)/CoFa.au_to_as
	A_diff = A_max - A_min

	adjusted_length_x = length_x - dist_to_border/2
	adjusted_length_y = length_y - dist_to_border/2

	anchor_A_x = -adjusted_length_x + abs(P.E_dir[0]*A_min)
	anchor_A_y =  adjusted_length_y - abs(A_max*P.E_dir[1])

	neg_A_x = np.array([anchor_A_x + A_min*P.E_dir[0], anchor_A_x])
	neg_A_y = np.array([anchor_A_y + A_min*P.E_dir[1], anchor_A_y])

	pos_A_x = np.array([anchor_A_x + A_max*P.E_dir[0], anchor_A_x])
	pos_A_y = np.array([anchor_A_y + A_max*P.E_dir[1], anchor_A_y])

	anchor_A_x_array = np.array([anchor_A_x])
	anchor_A_y_array = np.array([anchor_A_y])

	plt.plot(pos_A_x, pos_A_y, color="green")
	plt.plot(neg_A_x, neg_A_y, color="red")
	plt.plot(anchor_A_x_array, anchor_A_y_array, color='black', marker="o", linestyle='None')

	tikzplotlib.save("BZ.tikz", axis_height='\\figureheight', axis_width ='\\figurewidth' )
	plt.close()

	replace("scale=0.5",   "scale=1",	  filename="BZ.tikz")
	replace("mark size=3", "mark size=1", filename="BZ.tikz")
	replace("PH-SMALLNK1", str(Nk1_plot))
	replace("PH-SMALLNK2", str(Nk2_plot))
	replace("1.00000000000000000000",  str(ratio_yx))
	replace("figureheight,", "figureheight,	 scale only axis=true,", filename="BZ.tikz")

	class BZ_plot_parameters():
		pass

	K = BZ_plot_parameters()

	K.kx_BZ = kx_BZ
	K.ky_BZ = ky_BZ
	K.length_x = length_x
	K.length_y = length_y

	return K


def bandstruc_and_dipole_plot_high_symm_line(high_symmetry_path_BZ, P, num_points_for_plotting, sys):

	Nk1 = P.Nk1
	P.Nk1 = num_points_for_plotting

	path = high_symmetry_path_BZ

	sys.eigensystem_dipole_path(path, P)

	P.Nk1 = Nk1

	abs_k = np.sqrt(path[:,0]**2 + path[:,1]**2)

	k_in_path = np.zeros(num_points_for_plotting)

	for i_k in range(1,num_points_for_plotting):
		k_in_path[i_k] = k_in_path[i_k-1] + np.abs( abs_k[i_k] - abs_k[i_k-1] )

	_fig, (ax1) = plt.subplots(1)
	for i_band in range(P.n):
		_lines_exact_E_dir = ax1.plot(k_in_path, sys.e_in_path[:,i_band]*CoFa.au_to_eV, marker='', \
		                              label="$n=$ "+str(i_band))
	plot_it(P, r"Band energy " + unit['e(k)'], "bandstructure.tikz", ax1, k_in_path)
	plt.close(_fig)

	_fig, (ax2) = plt.subplots(1)
	d_min = 1.0E-10
	if P.dm_dynamics_method == 'semiclassics':
		for i_band in range(P.n):
			abs_connection = (np.sqrt( np.abs(sys.Ax_path[:, i_band, i_band])**2 + \
			                  np.abs(sys.Ay_path[:, i_band, i_band])**2 ) + 1.0e-80)*CoFa.au_to_as
			_lines_exact_E_dir= ax2.semilogy(k_in_path, abs_connection, marker='', \
			                                 label="$n=$ "+str(i_band))
			d_min = max(d_min, np.amin(abs_connection))
		plot_it(P, r'Berry connection ' + unit['dn'], "abs_dipole.tikz", ax2, k_in_path, d_min)

	else:
		for i_band in range(P.n):
			for j_band in range(P.n):
				if j_band >= i_band: continue
				abs_dipole = (np.sqrt(np.abs(sys.dipole_path_x[:, i_band, j_band])**2 + \
				              np.abs(sys.dipole_path_y[:, i_band, j_band])**2) + 1.0e-80)*CoFa.au_to_as
				_lines_exact_E_dir	= ax2.semilogy(k_in_path, abs_dipole, marker='', \
				                                   label="$n=$ "+str(i_band)+", $m=$ "+str(j_band))
				d_min = max(d_min, np.amin(abs_dipole))
		plot_it(P, r'Dipole ' + unit['dnm'], "abs_dipole.tikz", ax2, k_in_path, d_min)
	plt.close(_fig)


	_fig, (ax3) = plt.subplots(1)
	d_min = 1.0E-10
	if P.dm_dynamics_method == 'semiclassics':
		for i_band in range(P.n):
			proj_connection = (np.abs( sys.Ax_path[:,i_band,i_band]*P.E_dir[0] + \
			                   sys.Ay_path[:, i_band, i_band]*P.E_dir[1] ) + 1.0e-80)*CoFa.au_to_as
			_lines_exact_E_dir = ax3.semilogy(k_in_path, proj_connection, marker='',
			                                   label="$n=$ "+str(i_band))
			d_min = max(d_min, np.amin(proj_connection))
		plot_it(P, unit['ephi_dot_dn'], "proj_dipole.tikz", ax3, k_in_path, d_min)

	else:
		for i_band in range(P.n):
			for j_band in range(P.n):
				if j_band >= i_band: continue
				proj_dipole = (np.abs( sys.dipole_path_x[:,i_band,j_band]*P.E_dir[0] + \
				               sys.dipole_path_y[:,i_band,j_band]*P.E_dir[1] ) + 1.0e-80)/CoFa.au_to_as
				_lines_exact_E_dir = ax3.semilogy(k_in_path, proj_dipole, marker='', \
				                                  label="$n=$ "+str(i_band)+", $m=$ "+str(j_band))
				d_min = max(d_min, np.amin(proj_dipole))
		plot_it(P, unit['ephi_dot_dnm'], "proj_dipole.tikz", ax3, k_in_path, d_min)
	plt.close(_fig)

def plot_it(P, ylabel, filename, ax1, k_in_path, y_min=None):

	num_points_for_plotting = k_in_path.size
	k_lims = ( k_in_path[0], k_in_path[-1] )

	ax1.grid(True, axis='both', ls='--')
	ax1.set_ylabel(ylabel)
	ax1.legend(loc='upper left')
	ax1.set_xlim(k_lims)
	if y_min is not None:
		ax1.set_ylim(bottom=y_min)
	ax1.set_xticks( [k_in_path[0], k_in_path[num_points_for_plotting//2], k_in_path[-1]] )
	if P.BZ_type == 'hexagon':
		ax1.set_xticklabels([symb['K'], symb['G'], symb['M']])
	elif P.BZ_type == 'rectangle':
		ax1.set_xticklabels([symb['X'], symb['G'], symb['Y']])

	tikzplotlib.save(filename, axis_height='\\figureheight', axis_width ='\\figurewidth' )


def dipole_quiver_plots(K, P, sys):

	Nk1 = P.Nk1
	Nk2 = P.Nk2
	if P.BZ_type == 'rectangle':
		Nk_plot = 10
		P.Nk1	= Nk_plot
		P.Nk2	= Nk_plot
		length_BZ_E_dir	  = P.length_BZ_E_dir
		length_BZ_ortho	  = P.length_BZ_ortho
		P.length_BZ_E_dir = max(length_BZ_E_dir, length_BZ_ortho)
		P.length_BZ_ortho = max(length_BZ_E_dir, length_BZ_ortho)
	elif P.BZ_type == 'rectangle':
		P.Nk1 = 24
		P.Nk2 = 6

	Nk_combined = P.Nk1*P.Nk2

	d_x = np.zeros([Nk_combined, P.n, P.n], dtype=np.complex128)
	d_y = np.zeros([Nk_combined, P.n, P.n], dtype=np.complex128)
	k_x = np.zeros( Nk_combined )
	k_y = np.zeros( Nk_combined )

	if P.BZ_type == 'hexagon':
		dk, kweight, printed_paths, printed_mesh = hex_mesh(P)
	elif P.BZ_type == 'rectangle':
		dk, kweight, printed_paths, printed_mesh = rect_mesh(P)

	for k_path, path in enumerate(printed_paths):

		sys.eigensystem_dipole_path(path, P)

		d_x[k_path*P.Nk1:(k_path+1)*P.Nk1, :, :] = sys.dipole_path_x[:,:,:]*CoFa.au_to_as
		d_y[k_path*P.Nk1:(k_path+1)*P.Nk1, :, :] = sys.dipole_path_y[:,:,:]*CoFa.au_to_as
		k_x[k_path*P.Nk1:(k_path+1)*P.Nk1]		 = path[:,0]/CoFa.au_to_as
		k_y[k_path*P.Nk1:(k_path+1)*P.Nk1]		 = path[:,1]/CoFa.au_to_as

	num_plots = P.n**2
	num_plots_vert = (num_plots+1)//2

	fig, ax = plt.subplots(num_plots_vert, 2, figsize=(15, 6.2*num_plots_vert))

	for i_band in range(P.n):
		plot_x_index = i_band//2
		plot_y_index = i_band%2
		title = r"$\mb{{d}}_{{{:d}{:d}}}(\mb{{k}})$ (diagonal dipole matrix elements are real)"\
		    .format(i_band, i_band)
		colbar_title = r"$\log_{{10}}\; \lvert (\mb{{d}}_{{{:d}{:d}}}(\mb{{k}}))/\si{{\As}} \rvert$"\
		    .format(i_band, i_band)

		plot_single_dipole(d_x.real, d_y.real, i_band, i_band, plot_x_index, plot_y_index,
		                   K, k_x, k_y, fig, ax, title, colbar_title)

	counter = 0

	for i_band in range(P.n):
		for j_band in range(P.n):

			if i_band >= j_band: continue

			plot_index = P.n + counter
			plot_x_index = plot_index//2
			plot_y_index = plot_index%2
			counter += 1
			title = r"$\rRe \mb{{d}}_{{{:d}{:d}}}(\mb{{k}})$"\
			    .format(i_band, j_band)
			colbar_title = r"$\log_{{10}}\; \lvert \rRe(\mb{{d}}_{{{:d}{:d}}}(\mb{{k}}))/\si{{\As}} \rvert$"\
			    .format(i_band, j_band)
			plot_single_dipole(d_x.real, d_y.real, i_band, j_band, plot_x_index, plot_y_index, \
			                   K, k_x, k_y, fig, ax, title, colbar_title)

			plot_index = P.n + counter
			plot_x_index = plot_index//2
			plot_y_index = plot_index%2
			counter += 1
			title = r"$\rIm \mb{{d}}_{{{:d}{:d}}}(\mb{{k}})$"\
			    .format(i_band, j_band)
			colbar_title = r"$\log_{{10}}\; \lvert \rIm (\mb{{d}}_{{{:d}{:d}}}(\mb{{k}}))/\si{{\As}} \rvert$"\
			    .format(i_band, j_band)

			plot_single_dipole(d_x.imag, d_y.imag, i_band, j_band, plot_x_index, plot_y_index, \
			                   K, k_x, k_y, fig, ax, title, colbar_title)

	filename = 'dipoles.pdf'

	plt.savefig(filename, bbox_inches='tight')
	plt.close(fig)

	P.Nk1 = Nk1
	P.Nk2 = Nk2

	if P.BZ_type == 'rectangle':
		P.length_BZ_E_dir = length_BZ_E_dir
		P.length_BZ_ortho = length_BZ_ortho


def plot_single_dipole(d_x, d_y, i_band, j_band, x, y, K, k_x, k_y, fig, ax, \
                       title, colbar_title):

	d_x_ij = d_x[:, i_band, j_band]
	d_y_ij = d_y[:, i_band, j_band]

	abs_d_ij = np.maximum(np.sqrt(d_x_ij**2 + d_y_ij**2), 1.0e-10*np.ones(np.shape(d_x_ij)))

	norm_d_x_ij = d_x_ij / abs_d_ij
	norm_d_y_ij = d_y_ij / abs_d_ij

	ax[x,y].plot(K.kx_BZ, K.ky_BZ, color='gray' )
	plot = ax[x,y].quiver(k_x, k_y, norm_d_x_ij, norm_d_y_ij, np.log10(abs_d_ij),
	                      angles='xy', cmap=whitedarkjet, width=0.007 )

	ax[x,y].set_title(title)
	ax[x,y].axis('equal')
	ax[x,y].set_xlabel(unit['kx'])
	ax[x,y].set_ylabel(unit['ky'])
	ax[x,y].set_xlim(-K.length_x, K.length_x)
	ax[x,y].set_ylim(-K.length_y, K.length_y)

	plt.colorbar(plot, ax=ax[x,y], label=colbar_title)


def density_matrix_plot(P, T, K):
	i_band, j_band = 1, 1
	reshaped_pdf_dm = np.zeros((P.Nk1*P.Nk2, P.Nt_pdf_densmat, P.n, P.n), dtype=P.type_complex_np)

	for i_k1 in range(P.Nk1):
		for j_k2 in range(P.Nk2):
			combined_k_index = i_k1 + j_k2*P.Nk1
			reshaped_pdf_dm[combined_k_index, :, :, :] = T.pdf_densmat[i_k1, j_k2, :, :, :]

	n_vert = (P.Nt_pdf_densmat+1)//2
	for i_band in range(P.n):
		filename = 'dm_' + str(i_band) + str(i_band) + '.pdf'
		plot_dm_for_all_t(reshaped_pdf_dm.real, P, T, K, i_band, i_band, '', filename, n_vert)

	for i_band in range(P.n):
		for j_band in range(P.n):
			if i_band >= j_band: continue
			filename = 'Re_dm_' + str(i_band) + str(j_band) + '.pdf'
			plot_dm_for_all_t(reshaped_pdf_dm.real, P, T, K, i_band, j_band, 'Re', filename, n_vert)

			filename = 'Im_dm_' + str(i_band) + str(j_band) + '.pdf'
			plot_dm_for_all_t(reshaped_pdf_dm.imag, P, T, K, i_band, j_band, 'Im', filename, n_vert)

	replace("bandindex in {0,...,1}", "bandindex in {0,...," + str(P.n-1) + "}")


def plot_dm_for_all_t(reshaped_pdf_dm, P, T, K, i_band, j_band, prefix_title, \
                      filename, n_plots_vertical):

	fig, ax = plt.subplots(n_plots_vertical, 2, figsize=(15, 6.2*n_plots_vertical*K.length_y/K.length_x))

	for t_i in range(P.Nt_pdf_densmat):
		i = t_i//2
		j = t_i%2

		minval = np.amin(reshaped_pdf_dm[:, :, i_band, j_band].real)
		maxval = np.amax(reshaped_pdf_dm[:, :, i_band, j_band].real)
		if maxval - minval < 1E-6:
			minval -= 1E-6
			maxval += 1E-6

		if P.Nk2 > 1:
			im = ax[i, j].tricontourf(P.mesh[:, 0].astype('float64')/CoFa.au_to_as, P.mesh[:, 1].astype('float64')/CoFa.au_to_as,
			                       reshaped_pdf_dm[:, t_i, i_band, j_band].astype('float64'),
			                       np.linspace(minval, maxval, 100), cmap=whitedarkjet)

			# Aesthetics
			contourf_remove_white_lines(im)

			fig.colorbar(im, ax=ax[i, j])
			ax[i, j].plot(K.kx_BZ, K.ky_BZ, color='black')
			ax[i, j].set_ylim(-K.length_y, K.length_y)
			ax[i, j].set_ylabel(unit['ky'])

		else:
			im = ax[i,j].plot(P.mesh[:, 0]/CoFa.au_to_as, reshaped_pdf_dm[:, t_i, i_band, j_band])

		ax[i, j].set_xlabel(unit['kx'])
		ax[i, j].set_xlim(-K.length_x, K.length_x)
		ax[i, j].set_title(prefix_title +
		                   r' $\rho_{{{:d},{:d}}}(\mb{{k}},t)$ at $t = {:.1f}\si{{\fs}}$'\
		                   .format(i_band, j_band, T.t_pdf_densmat[t_i]*CoFa.au_to_fs))

	plt.savefig(filename, bbox_inches='tight')
	plt.close(fig)


def tikz_screening_one_color(S, num_points_for_plotting, title):
	'''
	Plot a screening (like CEP) plot in frequency range given by ff0 and screening_output
	'''
	# Find global min and max between 0th and 30th harmonic
	# Save all relevant outputs to plot in 3 horizontal plots
	num_subplots = 3
	fidx = np.empty(num_subplots, dtype=slice)
	I_min, I_max = np.empty(num_subplots, dtype=np.float64), np.empty(num_subplots, dtype=np.float64)
	screening_output = []
	freq_min = np.array([0, 10, 20])
	freq_max = freq_min + 10
	for i in range(freq_min.size):
		fidx[i] = get_freq_indices_for_plotting(S.ff0, num_points_for_plotting, freq_min[i], freq_max[i])
		screening_output.append(S.screening_output[:, fidx[i]])
		I_min[i], I_max[i] = screening_output[i].min(), screening_output[i].max()

	I_min, I_max = I_min.min(), I_max.max()
	S.I_max_in_plotting_range = '{:.4e}'.format(I_max)
	screening_output_norm = np.array(screening_output)/I_max
	I_min_norm, I_max_norm = I_min/I_max, 1
	I_min_norm_log, I_max_norm_log = np.log10(I_min_norm), np.log10(I_max_norm)


	fig, ax = plt.subplots(num_subplots)

	contourlevels = np.logspace(I_min_norm_log, I_max_norm_log, 1000)
	mintick, maxtick = int(I_min_norm_log), int(I_max_norm_log)
	tickposition = np.logspace(mintick, maxtick, num=np.abs(maxtick - mintick) + 1)

	cont = np.empty(num_subplots, dtype=object)

	for i, idx in enumerate(fidx):
		ff0 = S.ff0[idx]
		F, P = np.meshgrid(ff0, S.screening_parameter_values)
		cont[i] = ax[i].contourf(F, P, screening_output_norm[i], levels=contourlevels, locator=mpl.ticker.LogLocator(),
 		                         cmap=whitedarkjet, norm=mpl.colors.LogNorm(vmin=I_min_norm, vmax=I_max_norm))
		# Aesthetics
		contourf_remove_white_lines(cont[i])
		ax[i].set_xticks(np.arange(freq_min[i], freq_max[i] + 1))
		ax[i].set_ylabel(S.screening_parameter_name_plot_label)

	ax[-1].set_xlabel(r'Harmonic order ' + unit['ff0'])

	# Colorbar for ax[0]
	divider = make_axes_locatable(ax[0])
	cax = divider.append_axes('top', '7%', pad='2%')
	cbar = fig.colorbar(cont[0], cax=cax, orientation='horizontal')
	cax.tick_params(axis='x', which='major', top=True, pad=0.05)
	cax.xaxis.set_ticks_position('top')
	cax.invert_xaxis()
	cax.set_ylabel(r'$I_\mr{hh}/I_\mr{hh}^\mr{max}$', rotation='horizontal')
	cax.yaxis.set_label_coords(-0.1, 1.00)
	cbar.set_ticks(tickposition)
	# Disable every second tick label
	for label in cbar.ax.xaxis.get_ticklabels()[0::2]:
		label.set_visible(False)
	plt.suptitle(title)
	plt.savefig(S.screening_filename_plot, bbox_inches='tight')
	plt.close(fig)

def tikz_screening_per_color(S, num_points_for_plotting, title):
	'''
	Plot a screening (like CEP) plot in frequency range given by ff0 and screening_output
	'''
	# Find global min and max between 0th and 30th harmonic
	# Save all relevant outputs to plot in 3 horizontal plots
	num_subplots = 3
	fidx = np.empty(num_subplots, dtype=slice)
	I_min, I_max = np.empty(num_subplots, dtype=np.float64), np.empty(num_subplots, dtype=np.float64)
	screening_output = []
	freq_min = np.array([0, 10, 20])
	freq_max = freq_min + 10
	for i in range(freq_min.size):
		fidx[i] = get_freq_indices_for_plotting(S.ff0, num_points_for_plotting, freq_min[i], freq_max[i])
		screening_output.append(S.screening_output[:, fidx[i]])
		I_min[i], I_max[i] = screening_output[i].min(), screening_output[i].max()

	S.I_max_in_plotting_range = ['{:.4e}'.format(I_buf) for I_buf in I_max]
	screening_output_norm = np.array([screening_output[i]/I_max[i] for i in range(num_subplots)])
	I_min_norm, I_max_norm = I_min/I_max, np.ones(num_subplots)
	I_min_norm_log, I_max_norm_log = np.log10(I_min_norm), np.log10(I_max_norm)

	fig, ax = plt.subplots(num_subplots)
	contourlevels = np.logspace(I_min_norm_log, I_max_norm_log, 1000)
	mintick, maxtick = I_min_norm_log.astype(int), I_max_norm_log.astype(int)
	tickposition = [np.logspace(mintick[i], maxtick[i], num=np.abs(maxtick[i] - mintick[i]) + 1)
                  for i in range(num_subplots)]

	for i, idx in enumerate(fidx):
		ff0 = S.ff0[idx]
		F, P = np.meshgrid(ff0, S.screening_parameter_values)
		cont = ax[i].contourf(F, P, screening_output_norm[i], levels=contourlevels[:, i], locator=mpl.ticker.LogLocator(),
                          cmap=whitedarkjet, norm=mpl.colors.LogNorm(vmin=I_min_norm[i], vmax=I_max_norm[i]))
		# Per plot remove white lines
		contourf_remove_white_lines(cont)
		ax[i].set_xticks(np.arange(freq_min[i], freq_max[i] + 1))
		ax[i].set_ylabel(S.screening_parameter_name_plot_label)

		# Per plot axis label
		label_inner(ax[i], idx=i)

		# Per plot colorbar
		divider = make_axes_locatable(ax[i])
		cax = divider.append_axes('right', '7%', pad='2%')

		cbar = fig.colorbar(cont, cax=cax)
		cax.tick_params(axis='y', which='major', top=False, pad=0.05)
		# Set label only for first colorbar
		if i == 0:
			cax.set_ylabel(r'$I_\mr{hh}/I_\mr{hh}^\mr{max}$', rotation='horizontal')
			cax.yaxis.set_label_coords(1, 1.19)
		cbar.set_ticks(tickposition[i])

	ax[-1].set_xlabel(r'Harmonic order ' + unit['ff0'])
	ax[0].set_title(title)
	# Adjust plot name
	S.screening_filename = S.screening_filename + 'split_'
	plt.savefig(S.screening_filename_plot, bbox_inches='tight')
	plt.close(fig)

def write_and_compile_screening_latex_PDF(S):

	num_points_for_plotting = 960

	cued_copy('plotting/tex_templates/CUED_screening_summary.tex', '.')
	cued_copy('plotting/tex_templates/CUED_aliases.tex', '.')
	cued_copy('branding/logo.pdf', '.')

	# Sets I_max_in_plotting_range to single number
	tikz_screening_one_color(S[0], num_points_for_plotting, title='Intensity parallel to E-field direction')
	replace('PH-EDIR-PLOT', S[0].screening_filename_plot, filename="CUED_screening_summary.tex")
	replace('PH-EDIR-IMAX', S[0].I_max_in_plotting_range, filename="CUED_screening_summary.tex")
	replace('PH-PARAMETER', S[0].screening_parameter_name, filename="CUED_screening_summary.tex")

	# Sets I_max_in_plotting_range to list with 3 entries
	tikz_screening_per_color(S[0], num_points_for_plotting, title='Intensity parallel to E-field direction')
	replace('PH-EDIR-S-PLOT', S[0].screening_filename_plot, filename="CUED_screening_summary.tex")
	replace('PH-EDIR-A-IMAX', S[0].I_max_in_plotting_range[0], filename="CUED_screening_summary.tex")
	replace('PH-EDIR-B-IMAX', S[0].I_max_in_plotting_range[1], filename="CUED_screening_summary.tex")
	replace('PH-EDIR-C-IMAX', S[0].I_max_in_plotting_range[2], filename="CUED_screening_summary.tex")

	tikz_screening_one_color(S[1], num_points_for_plotting, title='Intensity orthogonal to E-field direction')
	replace('PH-ORTHO-PLOT', S[1].screening_filename_plot, filename="CUED_screening_summary.tex")
	replace('PH-ORTHO-IMAX', S[1].I_max_in_plotting_range, filename="CUED_screening_summary.tex")
	replace('PH-PARAMETER', S[1].screening_parameter_name, filename="CUED_screening_summary.tex")

	tikz_screening_per_color(S[1], num_points_for_plotting, title='Intensity orthogonal to E-field direction')
	replace('PH-ORTHO-S-PLOT', S[1].screening_filename_plot, filename="CUED_screening_summary.tex")
	replace('PH-ORTHO-A-IMAX', S[1].I_max_in_plotting_range[0], filename="CUED_screening_summary.tex")
	replace('PH-ORTHO-B-IMAX', S[1].I_max_in_plotting_range[1], filename="CUED_screening_summary.tex")
	replace('PH-ORTHO-C-IMAX', S[1].I_max_in_plotting_range[2], filename="CUED_screening_summary.tex")

	tikz_screening_one_color(S[2], num_points_for_plotting, title='Summed Intensity')
	replace('PH-FULL-PLOT', S[2].screening_filename_plot, filename="CUED_screening_summary.tex")
	replace('PH-FULL-IMAX', S[2].I_max_in_plotting_range, filename="CUED_screening_summary.tex")
	replace('PH-PARAMETER', S[2].screening_parameter_name, filename="CUED_screening_summary.tex")

	tikz_screening_per_color(S[2], num_points_for_plotting, title='Summed Intensity')
	replace('PH-FULL-S-PLOT', S[2].screening_filename_plot, filename="CUED_screening_summary.tex")
	replace('PH-FULL-A-IMAX', S[2].I_max_in_plotting_range[0], filename="CUED_screening_summary.tex")
	replace('PH-FULL-B-IMAX', S[2].I_max_in_plotting_range[1], filename="CUED_screening_summary.tex")
	replace('PH-FULL-C-IMAX', S[2].I_max_in_plotting_range[2], filename="CUED_screening_summary.tex")

	conditional_pdflatex(S[0].screening_filename.replace('_E_dir_split_', '').replace('_', ' '),
                       'CUED_screening_summary.tex')
