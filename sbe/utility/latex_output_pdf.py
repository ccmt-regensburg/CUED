import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from sbe.utility import ConversionFactors as co

def write_and_compile_latex_PDF(t, E_field, A_field, E_dir, paths, P):

        t_fs = t*co.au_to_fs

        t_idx = get_plot_limits_time(E_field, t_fs, factor_t_plot_end=1.5)

        latex_dir = "latex_pdf_files"

        if os.path.exists(latex_dir) and os.path.isdir(latex_dir):
            shutil.rmtree(latex_dir)

        os.mkdir(latex_dir)
        os.chdir(latex_dir)

        tikz_time(E_field*co.au_to_MVpcm, t_fs, t_idx, r'E-field $E(t)$ in MV/cm', "Efield")
        tikz_time(A_field*co.au_to_MVpcm*co.au_to_fs, t_fs, t_idx, r"A-field $A(t)$ in MV*fs/cm", "Afield")

        BZ_plot(paths, P, A_field, E_dir)



        code_path = os.path.dirname(os.path.realpath(__file__))

        shutil.copy(code_path+"/CUED_summary.tex", ".")
        shutil.copy(code_path+"/logo.pdf", ".")

        write_parameter(P)

        os.system("pdflatex CUED_summary.tex")

        os.system("pdflatex CUED_summary.tex")

        os.chdir("..")



def write_parameter(P):

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
    if P.phase == 0:
         replace("PH-CEP", 0)
    elif P.phase > np.pi/2-eps and P.phase < np.pi/2+eps:
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


def replace(old, new, filename="CUED_summary.tex"):

    print("sed -i -e \'s/"+old+"/"+new+"/g\' "+filename)

    os.system("sed -i -e \'s/"+old+"/"+new+"/g\' "+filename)


def get_plot_limits_time(E_field, time_fs, factor_t_plot_end): 

    E_max = np.amax(np.abs(E_field))

    threshold = 1.0E-3
  
    num_t_points_max = 200

    for i_counter, E_i in enumerate(E_field):
        if np.abs(E_i) > threshold*E_max: 
            index_t_plot_start = i_counter
            break

    for i_counter, E_i in reversed(list(enumerate(E_field))):
        if np.abs(E_i) > threshold*E_max: 
            t_plot_end       = time_fs[i_counter]
            break

    t_plot_end *= factor_t_plot_end

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


def BZ_plot(paths, P, A_field, E_dir):
    """
        Function that plots the Brillouin zone
    """
    BZ_fig = plt.figure(figsize=(10, 10))
    plt.plot(np.array([0.0]), np.array([0.0]), color='black', marker="o", linestyle='None')
    plt.text(0.01, 0.01, r'$\Gamma$')

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
        length = 5.0/P.a_angs

    elif P.BZ_type == 'rectangle':
        # polar angle of upper right point of a rectangle that is horizontally aligned
        alpha = np.arctan(P.length_BZ_ortho/P.length_BZ_E_dir)
        beta  = P.angle_inc_E_field/360*2*np.pi
        dist_edge_to_Gamma = np.sqrt(P.length_BZ_E_dir**2+P.length_BZ_ortho**2)
        rectangle_x = dist_edge_to_Gamma*np.array([np.cos(alpha+beta),np.cos(np.pi-alpha+beta),np.cos(alpha+beta+np.pi),np.cos(2*np.pi-alpha+beta),np.cos(alpha+beta)])
        rectangle_y = dist_edge_to_Gamma*np.array([np.sin(alpha+beta),np.sin(np.pi-alpha+beta),np.sin(alpha+beta+np.pi),np.sin(2*np.pi-alpha+beta),np.sin(alpha+beta)])
        plt.plot(rectangle_x, rectangle_y, color='black' )
        max_length = max(P.length_BZ_E_dir, P.length_BZ_ortho)
        length = 1.3*max_length

    plt.xlim(-length, length)
    plt.ylim(-length, length)

    plt.xlabel(r'$k_x$ in 1/Angstroem')
    plt.ylabel(r'$k_y$ in 1/Angstroem')

    for path in paths:
        num_k                = np.size(path[:,0])
        plot_path_x          = np.zeros(num_k+1)
        plot_path_y          = np.zeros(num_k+1)
        plot_path_x[0:num_k] = co.as_to_au*path[0:num_k, 0]
        plot_path_x[num_k]   = co.as_to_au*path[0, 0]
        plot_path_y[0:num_k] = co.as_to_au*path[0:num_k, 1]
        plot_path_y[num_k]   = co.as_to_au*path[0, 1]

        plt.plot(plot_path_x, plot_path_y)
        plt.plot(plot_path_x, plot_path_y, color='gray', marker="o", linestyle='None')

    A_min = np.amin(A_field)*co.as_to_au
    A_max = np.amax(A_field)*co.as_to_au
    A_diff = A_max - A_min

    dist_to_border = 0.05*length
    adjusted_length = length - dist_to_border

    neg_A_x = np.array([-adjusted_length,-adjusted_length-E_dir[0]*A_min])
    neg_A_y = np.array([adjusted_length-E_dir[1]*A_diff, adjusted_length-A_max*E_dir[1]])

    pos_A_x = np.array([-adjusted_length-E_dir[0]*A_min, -adjusted_length+E_dir[0]*A_diff])
    pos_A_y = np.array([adjusted_length-A_max*E_dir[1], adjusted_length])

    anchor_A_x = np.array([-adjusted_length-E_dir[0]*A_min])
    anchor_A_y = np.array([adjusted_length-A_max*E_dir[1]])

    plt.plot(pos_A_x, pos_A_y, color="green")
    plt.plot(neg_A_x, neg_A_y, color="red")
    plt.plot(anchor_A_x, anchor_A_y, color='black', marker="o", linestyle='None')

    tikzplotlib.save("BZ.tikz", axis_height='\\figureheight', axis_width ='\\figurewidth' )

    replace("scale=0.5",   "scale=1",     filename="BZ.tikz")
    replace("mark size=3", "mark size=1", filename="BZ.tikz")

#    plt.show()

