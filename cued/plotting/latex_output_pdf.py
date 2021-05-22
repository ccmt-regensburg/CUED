import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import tikzplotlib

from cued.utility import ConversionFactors as CoFa
from cued.kpoint_mesh import hex_mesh, rect_mesh

def write_and_compile_latex_PDF(T, W, P, sys, Mpi):

        t_fs = T.t*CoFa.au_to_fs
        num_points_for_plotting = 960

        t_idx = get_time_indices_for_plotting(T.E_field, t_fs, num_points_for_plotting)
        f_idx = get_freq_indices_for_plotting(W.freq/P.f, num_points_for_plotting, freq_max=30)

        t_idx_whole = get_indices_for_plotting_whole(t_fs, num_points_for_plotting, start=0)
        f_idx_whole = get_indices_for_plotting_whole(W.freq, num_points_for_plotting, start=f_idx[0])

        high_symmetry_path_BZ = get_symmetry_path_in_BZ(P, num_points_for_plotting)

        latex_dir = "latex_pdf_files"

        if os.path.exists(latex_dir) and os.path.isdir(latex_dir):
            shutil.rmtree(latex_dir)

        os.mkdir(latex_dir)
        os.chdir(latex_dir)

        code_path = os.path.dirname(os.path.realpath(__file__))

        shutil.copy(code_path+"/CUED_summary.tex", ".")
        shutil.copy(code_path+"/../branding/logo.pdf", ".")

        write_parameters(P, Mpi)

        tikz_time(T.E_field*CoFa.au_to_MVpcm, t_fs, t_idx, r'E-field $E(t)$ in MV/cm', "Efield")
        tikz_time(T.A_field*CoFa.au_to_MVpcm*CoFa.au_to_fs, t_fs, t_idx, r"A-field $A(t)$ in MV*fs/cm", "Afield")

        K = BZ_plot(P, T.A_field)

        bandstruc_and_dipole_plot_high_symm_line(high_symmetry_path_BZ, P, num_points_for_plotting, sys)

        dipole_quiver_plots(K, P, sys)

        density_matrix_plot(P, T, K)

        tikz_time(T.j_E_dir, t_fs, t_idx, \
                  r'Current $j_{\parallel}(t)$ parallel to $\bE$ in atomic units', "j_E_dir")
        tikz_time(T.j_E_dir, t_fs, t_idx_whole, \
                  r'Current $j_{\parallel}(t)$ parallel to $\bE$ in atomic units', "j_E_dir_whole_time")
        tikz_time(T.j_ortho, t_fs, t_idx, \
                  r'Current $j_{\bot}(t)$ orthogonal to $\bE$ in atomic units', "j_ortho")
        tikz_time(T.j_ortho, t_fs, t_idx_whole, \
                  r'Current $j_{\bot}(t)$ orthogonal to $\bE$ in atomic units', "j_ortho_whole_time")
        tikz_freq(W.I_E_dir, W.I_ortho, W.freq/P.f, f_idx_whole, \
                  r'Emission intensity in atomic units', "Emission_para_ortho_full_range", two_func=True, \
                  label_1="$\;I_{\parallel}(\omega)$", label_2="$\;I_{\\bot}(\omega)$")
        tikz_freq(W.I_E_dir, W.I_ortho, W.freq/P.f, f_idx, \
                  r'Emission intensity in atomic units', "Emission_para_ortho", two_func=True, \
                  label_1="$\;I_{\parallel}(\omega)$", label_2="$\;I_{\\bot}(\omega)$")
        tikz_freq(W.I_E_dir+W.I_ortho, None, W.freq/P.f, f_idx, \
                  r'Emission intensity in atomic units', "Emission_total", two_func=False, \
                  label_1="$\;I(\omega) = I_{\parallel}(\omega) + I_{\\bot}(\omega)$")
        tikz_freq(W.I_E_dir_hann+W.I_ortho_hann, W.I_E_dir_parzen+W.I_ortho_parzen, W.freq/P.f, f_idx, \
                  r'Emission intensity in atomic units', "Emission_total_hann_parzen", two_func=True, \
                  label_1="$\;I(\omega)$ with $\\bj(\omega)$ computed using the Hann window", \
                  label_2="$\;I(\omega)$ with $\\bj(\omega)$ computed using the Parzen window", dashed=True)

        replace("semithick", "thick", "*")

        os.system("pdflatex CUED_summary.tex")

        os.system("pdflatex CUED_summary.tex")

        os.chdir("..")


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
        replace("PH-E0",    str(P.E0_MVpcm))
        replace("PH-FREQ",  str(P.f_THz))
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

    xlabel = r'Time in fs'

    _fig, (ax1) = plt.subplots(1)
    _lines_exact_E_dir  = ax1.plot(time_fs[t_idx], func_of_t[t_idx], marker='')

    t_lims = (time_fs[t_idx[0]], time_fs[t_idx[-1]])
    
    ax1.grid(True, axis='both', ls='--')
    ax1.set_xlim(t_lims)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc='upper right')

    tikzplotlib.save(filename+".tikz",
                     axis_height='\\figureheight', 
                     axis_width ='\\figurewidth' )


def tikz_freq(func_1, func_2, freq_div_f0, f_idx, ylabel, filename, two_func, \
              label_1=None, label_2=None, dashed=False):

    xlabel = r'Harmonic order = (frequency $f$)/(pulse frequency $f_0$)'

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

    tikzplotlib.save(filename+".tikz",
                     axis_height='\\figureheight', 
                     axis_width ='\\figurewidth' )
 
    replace("xmax=30,", "xmax=30, xtick={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"+\
            ",21,22,23,24,25,26,27,28,29,30}, xticklabels={,1,,,,5,,,,,10,,,,,15,,,,,20,,,,,25,,,,,30},", \
            filename=filename+".tikz")


def replace(old, new, filename="CUED_summary.tex"):

    os.system("sed -i -e \'s/"+old+"/"+new+"/g\' "+filename)


def get_time_indices_for_plotting(E_field, time_fs, num_t_points_max): 

    E_max = np.amax(np.abs(E_field))

    threshold = 1.0E-3
  
    for i_counter, E_i in enumerate(E_field):
        if np.abs(E_i) > threshold*E_max: 
            index_t_plot_start = i_counter
            break

    for i_counter, E_i in reversed(list(enumerate(E_field))):
        if np.abs(E_i) > threshold*E_max: 
            index_t_plot_end = i_counter
            break

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


def get_freq_indices_for_plotting(freq_div_f0, num_points_for_plotting, freq_max):

    for i_counter, f_i in enumerate(freq_div_f0):
        if f_i.real > -1.0E-8: 
            index_f_plot_start = i_counter
            break

    for i_counter, f_i in enumerate(freq_div_f0):
        if f_i.real > freq_max:
            index_f_plot_end = i_counter
            break

    if index_f_plot_end - index_f_plot_start < num_points_for_plotting:
        step = 1
    else:
        step = (index_f_plot_end - index_f_plot_start)//num_points_for_plotting

    f_idx = range(index_f_plot_start, index_f_plot_end, step)

    return f_idx

def get_symmetry_path_in_BZ(P, num_points_for_plotting):

    Nk_per_line = num_points_for_plotting//2

    delta = 1/(2*Nk_per_line)
    neg_array_direct  = np.linspace(-0.5+delta,  0.0-delta, num=Nk_per_line)
    neg_array_reverse = np.linspace( 0.0+delta, -0.5-delta, num=Nk_per_line)
    pos_array_direct  = np.linspace( 0.0+delta,  0.5-delta, num=Nk_per_line)
    pos_array_reverse = np.linspace( 0.5-delta,  0.0+delta, num=Nk_per_line)

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
    plt.text(0.01, 0.01, r'$\Gamma$')
    default_width = 0.87

    if P.BZ_type == 'hexagon':
        R = 4.0*np.pi/(3*P.a_angs)
        r = 2.0*np.pi/(np.sqrt(3)*P.a_angs)
        plt.plot(np.array([r*np.cos(-np.pi/6)]), np.array([r*np.sin(-np.pi/6)]), color='black', marker="o", linestyle='None')
        plt.text(r*np.cos(-np.pi/6)+0.01, r*np.sin(-np.pi/6)-0.05, r'M')
        plt.plot(np.array([R]), np.array([0.0]), color='black', marker="o", linestyle='None')
        plt.text(R, 0.02, r'K')
        kx_BZ = R*np.array([1,2,1,0.5,1,    0.5,-0.5,-1,   -0.5,-1,-2,-1,-0.5,-1,    -0.5,0.5, 1,     0.5,  1])
        tmp = np.sqrt(3)/2
        ky_BZ = R*np.array([0,0,0,tmp,2*tmp,tmp,tmp, 2*tmp,tmp, 0, 0, 0, -tmp,-2*tmp,-tmp,-tmp,-2*tmp,-tmp,0])
        plt.plot(kx_BZ, ky_BZ, color='black' )
        length         = 5.0/P.a_angs
        length_x       = length
        length_y       = length
        ratio_yx       = default_width
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
        plt.text( X_x, X_y, r'X')
        plt.text( Y_x, Y_y, r'Y')

        dist_to_border = 0.1*max(np.amax(kx_BZ), np.amax(ky_BZ))
        length_x = np.amax(kx_BZ) + dist_to_border
        length_y = np.amax(ky_BZ) + dist_to_border
        ratio_yx = length_y/length_x*default_width

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

    plt.xlabel(r'$k_x$ in 1/\AA')
    plt.ylabel(r'$k_y$ in 1/\AA')

    for path in printed_paths:
        num_k                = np.size(path[:,0])
        plot_path_x          = np.zeros(num_k+1)
        plot_path_y          = np.zeros(num_k+1)
        plot_path_x[0:num_k] = 1/CoFa.au_to_as*path[0:num_k, 0]
        plot_path_x[num_k]   = 1/CoFa.au_to_as*path[0, 0]
        plot_path_y[0:num_k] = 1/CoFa.au_to_as*path[0:num_k, 1]
        plot_path_y[num_k]   = 1/CoFa.au_to_as*path[0, 1]

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

    replace("scale=0.5",   "scale=1",     filename="BZ.tikz")
    replace("mark size=3", "mark size=1", filename="BZ.tikz")
    replace("PH-SMALLNK1", str(Nk1_plot))
    replace("PH-SMALLNK2", str(Nk2_plot))
    replace("1.00000000000000000000",  str(ratio_yx))
    replace("figureheight,", "figureheight,  scale only axis=true,", filename="BZ.tikz")

    K = BZ_plot_parameters()

    K.kx_BZ = kx_BZ
    K.ky_BZ = ky_BZ
    K.length_x = length_x
    K.length_y = length_y

    return K

def bandstruc_and_dipole_plot_high_symm_line(high_symmetry_path_BZ, P, num_points_for_plotting, sys):

   Nk1    = P.Nk1
   P.Nk1  = num_points_for_plotting

   path = high_symmetry_path_BZ

   sys.eigensystem_dipole_path(path, P)

   P.Nk1 = Nk1

   abs_k  = np.sqrt(path[:,0]**2 + path[:,1]**2)

   k_in_path = np.zeros(num_points_for_plotting)

   for i_k in range(1,num_points_for_plotting):
       k_in_path[i_k] = k_in_path[i_k-1] + np.abs( abs_k[i_k] - abs_k[i_k-1] )

   _fig, (ax1) = plt.subplots(1)
   for i_band in range(P.n):
       _lines_exact_E_dir  = ax1.plot(k_in_path, sys.e_in_path[:,i_band]*CoFa.au_to_eV, marker='', \
                                      label="$n=$ "+str(i_band))
   plot_it(P,"Band energy $\epsilon_n(\mathbf{k})$ in eV", "bandstructure.tikz", ax1, k_in_path)

   _fig, (ax2) = plt.subplots(1)
   d_min = 1.0E-10
   for i_band in range(P.n):
       for j_band in range(P.n):
           if j_band >= i_band: continue
           abs_dipole = ( np.sqrt( np.abs(sys.dipole_path_x[:,i_band,j_band])**2 + \
                                   np.abs(sys.dipole_path_y[:,i_band,j_band])**2 ) + 1.0e-80)/CoFa.au_to_as
           _lines_exact_E_dir  = ax2.semilogy(k_in_path, abs_dipole, marker='', \
                                              label="$n=$ "+str(i_band)+", $m=$ "+str(j_band))
           d_min = max(d_min, np.amin(abs_dipole))
   plot_it(P,"Dipole $|\mathbf{d}_{nm}(\mathbf{k})|$ in 1/\AA","abs_dipole.tikz", ax2, k_in_path, d_min)


   _fig, (ax3) = plt.subplots(1)
   d_min = 1.0E-10
   for i_band in range(P.n):
       for j_band in range(P.n):
           if j_band >= i_band: continue
           proj_dipole = ( np.abs( sys.dipole_path_x[:,i_band,j_band]*P.E_dir[0] + \
                                   sys.dipole_path_y[:,i_band,j_band]*P.E_dir[1] ) + 1.0e-80)/CoFa.au_to_as
           _lines_exact_E_dir  = ax3.semilogy(k_in_path, proj_dipole, marker='', \
                                              label="$n=$ "+str(i_band)+", $m=$ "+str(j_band))
           d_min = max(d_min, np.amin(proj_dipole))
   plot_it(P,"$|\hat{e}_\phi\cdot\mathbf{d}_{nm}(\mathbf{k})|$ in 1/\AA","proj_dipole.tikz", ax3, k_in_path, d_min)


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
       ax1.set_xticklabels( ['K','$\Gamma$','M'] )
   elif P.BZ_type == 'rectangle':
       ax1.set_xticklabels( ['X','$\Gamma$','Y'] )

   tikzplotlib.save(filename,
                    axis_height='\\figureheight', 
                    axis_width ='\\figurewidth' )


def dipole_quiver_plots(K, P, sys):

    Nk1 = P.Nk1
    Nk2 = P.Nk2
    if P.BZ_type == 'rectangle':
        Nk_plot = 10
        P.Nk1   = Nk_plot
        P.Nk2   = Nk_plot
        length_BZ_E_dir   = P.length_BZ_E_dir
        length_BZ_ortho   = P.length_BZ_ortho
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
        k_x[k_path*P.Nk1:(k_path+1)*P.Nk1]       = path[:,0]/CoFa.au_to_as 
        k_y[k_path*P.Nk1:(k_path+1)*P.Nk1]       = path[:,1]/CoFa.au_to_as 

    num_plots = P.n**2
    num_plots_vert = (num_plots+1)//2

    fig, ax = plt.subplots(num_plots_vert, 2, figsize=(15,6.2*num_plots_vert))

    for i_band in range(P.n):
        plot_x_index = i_band//2
        plot_y_index = i_band%2
        title = r"$\mathbf{d}_{" + str(i_band) + str(i_band) + \
                "}(\mathbf{k})$ (diagonal dipole matrix elements are real)"
        colbar_title = r"log$_{10}\;|(\mathbf{d}_{"+str(i_band)+str(i_band)+"}(\mathbf{k}))/$\AA$|$"

        plot_single_dipole(d_x.real, d_y.real, i_band, i_band, plot_x_index, plot_y_index, \
                           K, k_x, k_y, fig, ax, title, colbar_title)

    counter = 0

    for i_band in range(P.n):
        for j_band in range(P.n):

            if i_band >= j_band: continue

            plot_index = P.n + counter
            plot_x_index = plot_index//2
            plot_y_index = plot_index%2
            counter += 1
            title = r"Re $\mathbf{d}_{" + str(i_band) + str(j_band) + "}(\mathbf{k})$"
            colbar_title = r"log$_{10}\;|\mathrm{Re } (\mathbf{d}_{"+str(i_band)+str(j_band)+\
                           "}(\mathbf{k}))/$\AA$|$"
    
            plot_single_dipole(d_x.real, d_y.real, i_band, j_band, plot_x_index, plot_y_index, \
                               K, k_x, k_y, fig, ax, title, colbar_title)

            plot_index = P.n + counter
            plot_x_index = plot_index//2
            plot_y_index = plot_index%2
            counter += 1
            title = r"Im $\mathbf{d}_{" + str(i_band) + str(j_band) + "}(\mathbf{k})$"
            colbar_title = r"log$_{10}\;|\mathrm{Im } (\mathbf{d}_{"+str(i_band)+str(j_band)+\
                           "}(\mathbf{k}))/$\AA$|$"

            plot_single_dipole(d_x.imag, d_y.imag, i_band, j_band, plot_x_index, plot_y_index, \
                               K, k_x, k_y, fig, ax, title, colbar_title)

    filename = 'dipoles.pdf'

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

    P.Nk1 = Nk1
    P.Nk2 = Nk2

    if P.BZ_type == 'rectangle':
        P.length_BZ_E_dir = length_BZ_E_dir
        P.length_BZ_ortho = length_BZ_ortho


def plot_single_dipole(d_x, d_y, i_band, j_band, x, y, K, k_x, k_y, fig, ax, \
                       title, colbar_title):

    d_x_ij = d_x[:, i_band, j_band]
    d_y_ij = d_y[:, i_band, j_band]

    abs_d_ij = np.maximum(np.sqrt( d_x_ij**2 + d_y_ij**2 ), 1.0e-10*np.ones(np.shape(d_x_ij)))

    norm_d_x_ij = d_x_ij / abs_d_ij
    norm_d_y_ij = d_y_ij / abs_d_ij

    ax[x,y].plot(K.kx_BZ, K.ky_BZ, color='gray' )
    plot = ax[x,y].quiver(k_x, k_y, norm_d_x_ij, norm_d_y_ij, np.log10(abs_d_ij),
                     angles='xy', cmap='coolwarm', width=0.007 )

    ax[x,y].set_title(title)
    ax[x,y].axis('equal')
    ax[x,y].set_xlabel(r'$k_x$ in 1/\AA')
    ax[x,y].set_ylabel(r'$k_y$ in 1/\AA')
    ax[x,y].set_xlim(-K.length_x, K.length_x)
    ax[x,y].set_ylim(-K.length_y, K.length_y)

    plt.colorbar(plot, ax=ax[x,y], label=colbar_title)


def density_matrix_plot(P, T, K):

    i_band = 1
    j_band = 1

    reshaped_pdf_dm = np.zeros((P.Nk1*P.Nk2, P.Nt_pdf_densmat, P.n, P.n), dtype=P.type_complex_np)

    for i_k1 in range(P.Nk1):
        for j_k2 in range(P.Nk2):

            combined_k_index = i_k1 + j_k2*P.Nk1
            reshaped_pdf_dm[combined_k_index, :, :, :] = T.pdf_densmat[i_k1, j_k2, :, :, :]

    n_vert = (P.Nt_pdf_densmat+1)//2

    for i_band in range(P.n):

        filename = 'dm_'+str(i_band)+str(i_band)+'.pdf'

        plot_dm_for_all_t(reshaped_pdf_dm.real, P, T, K, i_band, i_band, '', filename, n_vert)

    for i_band in range(P.n):

        for j_band in range(P.n):

            if i_band >= j_band: continue

            filename = 'Re_dm_'+str(i_band)+str(j_band)+'.pdf'
            plot_dm_for_all_t(reshaped_pdf_dm.real, P, T, K, i_band, j_band, 'Re', filename, n_vert)

            filename = 'Im_dm_'+str(i_band)+str(j_band)+'.pdf'
            plot_dm_for_all_t(reshaped_pdf_dm.imag, P, T, K, i_band, j_band, 'Im', filename, n_vert)


    replace("bandindex in {0,...,1}", "bandindex in {0,...,"+str(P.n-1)+"}")


def plot_dm_for_all_t(reshaped_pdf_dm, P, T, K, i_band, j_band, prefix_title, \
                                  filename, n_plots_vertical):

    fig, ax = plt.subplots(n_plots_vertical, 2, figsize=(15,6.2*n_plots_vertical*K.length_y/K.length_x))

    for t_i in range(P.Nt_pdf_densmat):

        i = t_i//2
        j = t_i%2

        minval = np.amin(reshaped_pdf_dm[:, :, i_band, j_band].real)
        maxval = np.amax(reshaped_pdf_dm[:, :, i_band, j_band].real)
        if maxval - minval < 1E-6:
            minval -= 1E-6
            maxval += 1E-6

        step = (maxval-minval)/100

        if P.Nk2 > 1:

            im = ax[i,j].tricontourf(P.mesh[:,0]/CoFa.au_to_as, P.mesh[:,1]/CoFa.au_to_as, \
                            reshaped_pdf_dm[:, t_i, i_band, j_band] , \
                            np.arange(minval,maxval,step), cmap='nipy_spectral')

            fig.colorbar(im, ax=ax[i,j])

            ax[i,j].plot(K.kx_BZ, K.ky_BZ, color='black')
            ax[i,j].set_ylim(-K.length_y, K.length_y)
            ax[i,j].set_ylabel(r'$k_y$ in 1/\AA')

        else:

            im = ax[i,j].plot(P.mesh[:,0]/CoFa.au_to_as, reshaped_pdf_dm[:, t_i, i_band, j_band])

        ax[i,j].set_xlabel(r'$k_x$ in 1/\AA')
        ax[i,j].set_xlim(-K.length_x, K.length_x)
        ax[i,j].set_title(prefix_title+' $\\rho_{'+str(i_band)+','+str(j_band)+'}(\mathbf{k},t)$ at $t ='+ \
                          '{:.1f}'.format(T.t_pdf_densmat[t_i]*CoFa.au_to_fs) + '$ fs')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


class BZ_plot_parameters():
    pass
