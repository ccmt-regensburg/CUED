import matplotlib as mpl

symb = {'G': r'$\Gamma$',
        'K': r'$\mathrm{K}$',
        'M': r'$\mathrm{M}$',
        'X': r'$\mathrm{X}$',
        'Y': r'$\mathrm{Y}$'}

unit = {# Band energy: electronvolts
        'e(k)': r'$\epsilon_n(\mathbf{k}) \; (\si{\electronvolt})$',
        # Time: femtoseconds
        't': r'$t \; (\si{\femto\second})$',
        # k_x: 1/angström
        'kx': r'$k_x \; (\si{\per\angstrom})$',
        # k_y: 1/angström
        'ky': r'$k_y \; (\si{\per\angstrom})$',
        # Berry Connection: e * angström
        'dn': r'$\lvert \mathbf{A}_{n}(\mathbf{k}) \rvert \; (\si{\elementarycharge\angstrom})',
        # Dipole Moment: e * angström
        'dnm': r'$\lvert \mathbf{d}_{nm}(\mathbf{k}) \rvert \; (\si{\elementarycharge\angstrom})',
        # Projected Berry Connection: e * angström
        'ephi_dot_dn': r'$\lvert \hat{e}_\phi \cdot \mathbf{A}_{n}(\mathbf{k}) \rvert \; (\si{\elementarycharge\angstrom})',
        # Projected Dipole Moment: e * angström
        'ephi_dot_dnm': r'$\lvert \hat{e}_\phi \cdot \mathbf{d}_{nm}(\mathbf{k}) \rvert \; (\si{\elementarycharge\angstrom})',
        # Frequency: terahertz
        'f': r'$f \; (\si{\tera\hertz})$',
        # Harmonic Order
        'ff0': r'$f/f_0$',
        # Electric Field: megavolt/cm
        'E': r'$\mathrm{E}(t) \; (\si{\mega\volt\per\cm})$',
        # Vector Potential: megavolt * femtoseconds/cm
        'A': r'$\mathrm{A}(t) \; (\si{\mega\volt\per\cm\fs})$',
        'A_fourier': r'$\mathrm{A}(\omega) \; (\si{\mega\volt\per\cm\femto\second^2})$',
        'eim_fourier': r'$\mathrm{Im}[\mathrm{E}(\omega)] \; (\si{\mega\volt\per\cm\femto\second})$'}

def init_matplotlib_config():
	# LAYOUT
	mpl.rc('figure', figsize=(5.295, 6.0), autolayout=True)
	# LATEX
	mpl.rc('text', usetex=True)
	mpl.rc('text.latex',
	       preamble=r'\usepackage{siunitx}\usepackage{braket}\usepackage{mathtools}\usepackage{amssymb}\usepackage[version=4]{mhchem}\usepackage[super]{nth}')#\sisetup{per-mode=symbol-or-fraction}')
	# FONT
	mpl.rc('font', size=10, family='serif', serif='CMU Serif')
	# mpl.rc('mathtext', fontset='cm')
	# AXES
	mpl.rc('axes', titlesize=10, labelsize=10)
	# LEGEND
	mpl.rc('legend', fancybox=False, fontsize=10, framealpha=1, labelspacing=0.08,
	       borderaxespad=0.05)
	# PGF
	mpl.rc('pgf', texsystem='lualatex',
	       preamble=r'\usepackage{siunitx}\usepackage{braket}\usepackage{mathtools}\usepackage{amssymb}\usepackage[version=4]{mhchem}\usepackage[super]{nth}')#\sisetup{per-mode=symbol-or-fraction}')
	
	mpl.use('Agg')

