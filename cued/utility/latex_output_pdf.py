import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from cued.utility import ConversionFactors as co
from cued.kpoint_mesh import hex_mesh, rect_mesh

def write_and_compile_latex_PDF(t, freq, E_field, A_field, I_exact_E_dir, I_exact_ortho, \
        Int_exact_E_dir, Int_exact_ortho, E_dir, paths, run_time, P, S):

        t_fs = t*co.au_to_fs
        num_points_max_for_plotting = 1000
        t_idx = get_time_indices_for_plotting(E_field, t_fs, num_points_max_for_plotting, factor_t_end=1.0)
        f_idx = get_freq_indices_for_plotting(freq/P.w, num_points_max_for_plotting, freq_max=30)

        latex_dir = "latex_pdf_files"

        if os.path.exists(latex_dir) and os.path.isdir(latex_dir):
            shutil.rmtree(latex_dir)

        os.mkdir(latex_dir)
        os.chdir(latex_dir)

        code_path = os.path.dirname(os.path.realpath(__file__))

        shutil.copy(code_path+"/CUED_summary.tex", ".")
        shutil.copy(code_path+"/logo.pdf", ".")

        write_parameter(P, run_time)

        tikz_time(E_field*co.au_to_MVpcm, t_fs, t_idx, r'E-field $E(t)$ in MV/cm', "Efield")
        tikz_time(A_field*co.au_to_MVpcm*co.au_to_fs, t_fs, t_idx, r"A-field $A(t)$ in MV*fs/cm", "Afield")

        BZ_plot(paths, P, A_field, S)

        tikz_time(I_exact_E_dir, t_fs, t_idx, \
                  r'Current $j_{\parallel}(t)$ parallel to $\bE$ in atomic units', "j_E_dir")
        tikz_time(I_exact_ortho, t_fs, t_idx, \
                  r'Current $j_{\bot}(t)$ orthogonal to $\bE$ in atomic units', "j_ortho")
        tikz_freq(Int_exact_E_dir, Int_exact_ortho, freq/P.w, f_idx, \
                  r'Emission intensity in atomic units', "Emission_total", two_func=True, \
                  label_1="$\;I_{\parallel}(\omega)$", label_2="$\;I_{\\bot}(\omega)$")
        tikz_freq(Int_exact_E_dir+Int_exact_ortho, None, freq/P.w, f_idx, \
                  r'Emission intensity in atomic units', "Emission_para_ortho", two_func=False, \
                  label_1="$\;I(\omega) = I_{\parallel}(\omega) + I_{\\bot}(\omega)$")
        replace("semithick", "thick", "*")

        os.system("pdflatex CUED_summary.tex")

        os.system("pdflatex CUED_summary.tex")

        os.chdir("..")


def write_parameter(P, run_time):

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

    replace("PH-FREQ",  str(P.w_THz))
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

    replace("PH-ALPHA", str(P.alpha_fs))
    replace("PH-FWHM", '{:.3f}'.format(P.alpha_fs*4*np.sqrt(np.log(2))))
    replace("PH-BZ", P.BZ_type)
    replace("PH-NK1", str(P.Nk1))
    replace("PH-NK2", str(P.Nk2))
    replace("PH-T2", str(P.T2_fs))
    replace("PH-RUN", '{:.1f}'.format(run_time))


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


def tikz_freq(func_of_f_1, func_of_f_2, freq_normalized, f_idx, ylabel, filename, two_func, \
              label_1=None, label_2=None):

    xlabel = r'Harmonic order = (frequency $f$)/(pulse frequency $f_0$)'

    _fig, (ax1) = plt.subplots(1)
    _lines_exact_E_dir = ax1.semilogy(freq_normalized[f_idx], func_of_f_1[f_idx], marker='', label=label_1)
    if two_func:
       _lines_exact_E_dir = ax1.semilogy(freq_normalized[f_idx], func_of_f_2[f_idx], marker='', label=label_2)

    f_lims = (freq_normalized[f_idx[0]], freq_normalized[f_idx[-1]])
    
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


def get_freq_indices_for_plotting(freq_normalized, num_points_max_for_plotting, freq_max):

    for i_counter, f_i in enumerate(freq_normalized):
        if f_i.real > -1.0E-8: 
            index_f_plot_start = i_counter
            break

    for i_counter, f_i in enumerate(freq_normalized):
        if f_i.real > freq_max:
            index_f_plot_end = i_counter
            break

    if index_f_plot_end - index_f_plot_start < num_points_max_for_plotting:
        step = 1
    else:
        step = (index_f_plot_end - index_f_plot_start)//num_points_max_for_plotting

    f_idx = range(index_f_plot_start, index_f_plot_end, step)

    return f_idx


def BZ_plot(paths, P, A_field, S):
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
        hexagon_x = R*np.array([1,2,1,0.5,1,    0.5,-0.5,-1,   -0.5,-1,-2,-1,-0.5,-1,    -0.5,0.5, 1,     0.5,  1])
        tmp = np.sqrt(3)/2
        hexagon_y = R*np.array([0,0,0,tmp,2*tmp,tmp,tmp, 2*tmp,tmp, 0, 0, 0, -tmp,-2*tmp,-tmp,-tmp,-2*tmp,-tmp,0])
        plt.plot(hexagon_x, hexagon_y, color='black' )
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
        rectangle_x = dist_edge_to_Gamma*np.array([np.cos(alpha+beta),np.cos(np.pi-alpha+beta),np.cos(alpha+beta+np.pi),np.cos(2*np.pi-alpha+beta),np.cos(alpha+beta)])
        rectangle_y = dist_edge_to_Gamma*np.array([np.sin(alpha+beta),np.sin(np.pi-alpha+beta),np.sin(alpha+beta+np.pi),np.sin(2*np.pi-alpha+beta),np.sin(alpha+beta)])
        plt.plot(rectangle_x, rectangle_y, color='black' )
        dist_to_border = 0.1*max(np.amax(rectangle_x), np.amax(rectangle_y))
        length_x = np.amax(rectangle_x) + dist_to_border
        length_y = np.amax(rectangle_y) + dist_to_border
        ratio_yx = length_y/length_x*default_width

    Nk1_max = 24
    Nk2_max = 6
    if P.Nk1 <= Nk1_max and P.Nk2 <= Nk2_max:
        printed_paths = paths
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

    plt.xlabel(r'$k_x$ in 1/Angstroem')
    plt.ylabel(r'$k_y$ in 1/Angstroem')

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


#    plt.show()

