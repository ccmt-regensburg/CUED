import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from cued.utility import ConversionFactors as co
from cued.utility import system_properties
from cued.kpoint_mesh import hex_mesh, rect_mesh
from cued.dipole import calculate_system_in_path

def write_and_compile_latex_PDF(T, W, P, S):

        t_fs = T.t*co.au_to_fs
        num_points_for_plotting = 960

        t_idx = get_time_indices_for_plotting(T.E_field, t_fs, num_points_for_plotting, factor_t_end=1.0)
        f_idx = get_freq_indices_for_plotting(W.freq/P.f, num_points_for_plotting, freq_max=30)

        t_idx_whole = get_indices_for_plotting_whole(t_fs, num_points_for_plotting, start=0)
        f_idx_whole = get_indices_for_plotting_whole(W.freq, num_points_for_plotting, start=f_idx[0])

        high_symmetry_path_BZ = get_symmetry_path_in_BZ(P, S, num_points_for_plotting)

        latex_dir = "latex_pdf_files"

        if os.path.exists(latex_dir) and os.path.isdir(latex_dir):
            shutil.rmtree(latex_dir)

        os.mkdir(latex_dir)
        os.chdir(latex_dir)

        code_path = os.path.dirname(os.path.realpath(__file__))

        shutil.copy(code_path+"/CUED_summary.tex", ".")
        shutil.copy(code_path+"/logo.pdf", ".")

        write_parameter(P, S)

        tikz_time(T.E_field*co.au_to_MVpcm, t_fs, t_idx, r'E-field $E(t)$ in MV/cm', "Efield")
        tikz_time(T.A_field*co.au_to_MVpcm*co.au_to_fs, t_fs, t_idx, r"A-field $A(t)$ in MV*fs/cm", "Afield")

        kx_BZ, ky_BZ = BZ_plot(P, T.A_field, S)

        bandstruc_and_dipole_plot_high_symm_line(high_symmetry_path_BZ, P, S, num_points_for_plotting)

#        dipole_quiver_plots(kx_BZ, ky_BZ, P, S)

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


def write_parameter(P, S):

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
    replace("PH-RUN", '{:.1f}'.format(S.run_time))
    replace("PH-MPIRANKS", str(S.Mpi.size))


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


def get_time_indices_for_plotting(E_field, time_fs, num_t_points_max, factor_t_end): 

    E_max = np.amax(np.abs(E_field))

    threshold = 1.0E-3
  
    for i_counter, E_i in enumerate(E_field):
        if np.abs(E_i) > threshold*E_max: 
            index_t_plot_start = i_counter
            break

    for i_counter, E_i in reversed(list(enumerate(E_field))):
        if np.abs(E_i) > threshold*E_max: 
            t_plot_end       = time_fs[i_counter]
            break

    t_plot_end *= factor_t_end

    for i_counter, t_i in enumerate(time_fs):
        if t_i > t_plot_end: 
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

def get_symmetry_path_in_BZ(P, S, num_points_for_plotting):

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

         vec_k_E_dir = P.length_BZ_E_dir*S.E_dir
         vec_k_ortho = P.length_BZ_ortho*np.array([S.E_dir[1], -S.E_dir[0]])
     
         path = []
         for alpha in pos_array_reverse:
             kpoint = alpha*vec_k_E_dir
             path.append(kpoint)

         for alpha in pos_array_direct:
             kpoint = alpha*vec_k_ortho
             path.append(kpoint)

    return np.array(path)


def BZ_plot(P, A_field, S):
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
        dist_edge_to_Gamma = np.sqrt(P.length_BZ_E_dir**2+P.length_BZ_ortho**2)/2/co.au_to_as
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
        printed_paths = S.paths
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
            dk, kweight, printed_paths = hex_mesh(P)
        elif P.BZ_type == 'rectangle':
            dk, kweight, printed_paths = rect_mesh(P, S)
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
        plot_path_x[0:num_k] = 1/co.au_to_as*path[0:num_k, 0]
        plot_path_x[num_k]   = 1/co.au_to_as*path[0, 0]
        plot_path_y[0:num_k] = 1/co.au_to_as*path[0:num_k, 1]
        plot_path_y[num_k]   = 1/co.au_to_as*path[0, 1]

        plt.plot(plot_path_x, plot_path_y)
        plt.plot(plot_path_x, plot_path_y, color='gray', marker="o", linestyle='None')

    A_min = np.amin(A_field)/co.au_to_as
    A_max = np.amax(A_field)/co.au_to_as
    A_diff = A_max - A_min

    adjusted_length_x = length_x - dist_to_border/2
    adjusted_length_y = length_y - dist_to_border/2

    anchor_A_x = -adjusted_length_x + abs(S.E_dir[0]*A_min)
    anchor_A_y =  adjusted_length_y - abs(A_max*S.E_dir[1])

    neg_A_x = np.array([anchor_A_x + A_min*S.E_dir[0], anchor_A_x])
    neg_A_y = np.array([anchor_A_y + A_min*S.E_dir[1], anchor_A_y])

    pos_A_x = np.array([anchor_A_x + A_max*S.E_dir[0], anchor_A_x])
    pos_A_y = np.array([anchor_A_y + A_max*S.E_dir[1], anchor_A_y])

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


    return kx_BZ, ky_BZ

def bandstruc_and_dipole_plot_high_symm_line(high_symmetry_path_BZ, P, S, num_points_for_plotting):

   Nk1    = P.Nk1
   P.Nk1  = num_points_for_plotting
   S_tmp  = system_properties(P, S.sys)

   path = high_symmetry_path_BZ

   calculate_system_in_path(path, P, S_tmp)

   P.Nk1 = Nk1

   abs_k  = np.sqrt(path[:,0]**2 + path[:,1]**2)

   k_in_path = np.zeros(num_points_for_plotting)

   for i_k in range(1,num_points_for_plotting):
       k_in_path[i_k] = k_in_path[i_k-1] + np.abs( abs_k[i_k] - abs_k[i_k-1] )

   _fig, (ax1) = plt.subplots(1)
   for i_band in range(P.n):
       _lines_exact_E_dir  = ax1.plot(k_in_path, S_tmp.e_in_path[:,i_band]*co.au_to_eV, marker='', \
                                      label="$n=$ "+str(i_band))
   plot_it(P,"Band energy $\epsilon_n(\mathbf{k})$ in eV", "bandstructure.tikz", ax1, k_in_path)

   _fig, (ax2) = plt.subplots(1)
   for i_band in range(P.n):
       for j_band in range(P.n):
           if j_band >= i_band: continue
           abs_dipole = np.sqrt( np.abs(S_tmp.dipole_path_x[:,i_band,j_band])**2 + \
                                 np.abs(S_tmp.dipole_path_y[:,i_band,j_band])**2 )/co.au_to_as
           _lines_exact_E_dir  = ax2.semilogy(k_in_path, abs_dipole, marker='', \
                                              label="$n=$ "+str(i_band)+", $m=$ "+str(j_band))
   plot_it(P,"Dipole $|\mathbf{d}_{nm}(\mathbf{k})|$ in 1/\AA","abs_dipole.tikz", ax2, k_in_path)


   _fig, (ax3) = plt.subplots(1)
   for i_band in range(P.n):
       for j_band in range(P.n):
           if j_band >= i_band: continue
           proj_dipole = np.abs( S_tmp.dipole_path_x[:,i_band,j_band]*S.E_dir[0] + \
                                 S_tmp.dipole_path_y[:,i_band,j_band]*S.E_dir[1] )/co.au_to_as
           _lines_exact_E_dir  = ax3.semilogy(k_in_path, proj_dipole, marker='', \
                                              label="$n=$ "+str(i_band)+", $m=$ "+str(j_band))
   plot_it(P,"$|\hat{e}_\phi\cdot\mathbf{d}_{nm}(\mathbf{k})|$ in 1/\AA","proj_dipole.tikz", ax3, k_in_path)


def plot_it(P, ylabel, filename, ax1, k_in_path):

   num_points_for_plotting = k_in_path.size
   k_lims = ( k_in_path[0], k_in_path[-1] )

   ax1.grid(True, axis='both', ls='--')
   ax1.set_ylabel(ylabel)
   ax1.legend(loc='upper left')
   ax1.set_xlim(k_lims)
   ax1.set_xticks( [k_in_path[0], k_in_path[num_points_for_plotting//2], k_in_path[-1]] )
   if P.BZ_type == 'hexagon':
       ax1.set_xticklabels( ['K','$\Gamma$','M'] )
   elif P.BZ_type == 'rectangle':
       ax1.set_xticklabels( ['X','$\Gamma$','Y'] )

   tikzplotlib.save(filename,
                    axis_height='\\figureheight', 
                    axis_width ='\\figurewidth' )


def dipole_quiver_plots(kx_BZ, ky_BZ, P, S):

    Nk_plot = 10
    Nk1     = P.Nk1
    Nk2     = P.Nk2
    P.Nk1   = Nk_plot
    P.Nk2   = Nk_plot
    if P.BZ_type == 'rectangle':
        length_BZ_E_dir   = P.length_BZ_E_dir
        length_BZ_ortho   = P.length_BZ_ortho
        P.length_BZ_E_dir = max(length_BZ_E_dir, length_BZ_ortho)
        P.length_BZ_ortho = max(length_BZ_E_dir, length_BZ_ortho)
 
    S_tmp  = system_properties(P, S.sys)
 
    P.Nk1             = Nk1
    P.Nk2             = Nk2
    P.length_BZ_E_dir = length_BZ_E_dir
    P.length_BZ_ortho = length_BZ_ortho
 
    d_x = np.zeros([Nk_plot**2, P.n, P.n], dtype=np.complex128)
    d_y = np.zeros([Nk_plot**2, P.n, P.n], dtype=np.complex128)
    k_x = np.zeros( Nk_plot**2 )
    k_y = np.zeros( Nk_plot**2 )

    for k_path, path in enumerate(S_tmp.paths):

        calculate_system_in_path(path, P, S_tmp)

        d_x[k_path*Nk_plot:(k_path+1)*Nk_plot, :, :] = S_tmp.dipole_path_x[:,:,:]*co.au_to_as 
        d_y[k_path*Nk_plot:(k_path+1)*Nk_plot, :, :] = S_tmp.dipole_path_y[:,:,:]*co.au_to_as 
        k_x[k_path*Nk_plot:(k_path+1)*Nk_plot]       = path[:,0]/co.au_to_as 
        k_y[k_path*Nk_plot:(k_path+1)*Nk_plot]       = path[:,1]/co.au_to_as 

    for i_band in range(P.n):
        for j_band in range(P.n):

            Re_d_x = np.real(d_x[:,i_band,j_band]) 
            Re_d_y = np.real(d_y[:,i_band,j_band])
            Im_d_x = np.imag(d_x[:,i_band,j_band])
            Im_d_y = np.imag(d_y[:,i_band,j_band])

            abs_Re_d = np.maximum(np.sqrt( Re_d_x**2 + Re_d_y**2 ), 1.0e-32*np.ones(np.shape(Re_d_x)))
            abs_Im_d = np.maximum(np.sqrt( Im_d_x**2 + Im_d_y**2 ), 1.0e-32*np.ones(np.shape(Im_d_x)))

            norm_Re_d_x = Re_d_x / abs_Re_d
            norm_Re_d_y = Re_d_y / abs_Re_d
            norm_Im_d_x = Im_d_x / abs_Im_d
            norm_Im_d_y = Im_d_y / abs_Im_d

            ij_index = i_band*P.n + j_band

            fig, ax = plt.subplots(1)
            ax.plot(kx_BZ, ky_BZ, color='gray' )
            plot = ax.quiver(k_x, k_y, norm_Re_d_x, norm_Re_d_y, np.log10(abs_Re_d),
                                          angles='xy', cmap='coolwarm', width=0.007 )

            current_name = r"Re $(\mathbf{d}_{" + str(i_band) + str(j_band) + "})$"
            current_abs_name = r"log$_{10}\;| \mathrm{Re} (\mathbf{d}_{" + str(i_band) + str(j_band) + "})|$"
            ax.set_title(current_name)
            ax.axis('equal')
            ax.set_xlabel(r'$k_x$ in 1/\AA')
            ax.set_ylabel(r'$k_y$ in 1/\AA')
            plt.colorbar(plot, ax=ax, label=current_abs_name)
#            tikzplotlib.save("dipole_quiver_Re_d_"+str(i_band)+"_"+str(j_band)+".tikz", 
#                             axis_height='\\figureheight', axis_width ='\\figurewidth' )

            fig, ax = plt.subplots(1)
            ax.plot(kx_BZ, ky_BZ, color='gray' )
            plot = ax.quiver(k_x, k_y, norm_Im_d_x, norm_Im_d_y, np.log10(abs_Im_d),
                             angles='xy', cmap='coolwarm', width=0.007 )

            current_name = r"Im $(\mathbf{d}_{" + str(i_band) + str(j_band) + "})$"
            current_abs_name = r"log$_{10}\;| \mathrm{Im} (\mathbf{d}_{" + str(i_band) + str(j_band) + "})|$"
            ax.set_title(current_name)
            ax.axis('equal')
            ax.set_xlabel(r'$k_x$ in 1/\AA')
            ax.set_ylabel(r'$k_y$ in 1/\AA')
            plt.colorbar(plot, ax=ax, label=current_abs_name)
#            tikzplotlib.save("dipole_quiver_Im_d_"+str(i_band)+"_"+str(j_band)+".tikz", 
#                             axis_height='\\figureheight', axis_width ='\\figurewidth' )


    plt.show()
