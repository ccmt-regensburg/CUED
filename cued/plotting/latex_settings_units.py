import matplotlib as mpl
from cued.utility import cued_pwd

symb = {'G': r'$\Gamma$',
        'K': r'$\mr{K}$',
        'M': r'$\mr{M}$',
        'X': r'$\mr{X}$',
        'Y': r'$\mr{Y}$'}

unit = {# Band energy: electronvolts
        'e(k)': r'$\epsilon_n(\mb{k}) \; (\si{\electronvolt})$',
        # Time: femtoseconds
        't': r'$t \; (\si{\femto\second})$',
        # k_x: 1/angström
        'kx': r'$k_x \; (\si{\per\angstrom})$',
        # k_y: 1/angström
        'ky': r'$k_y \; (\si{\per\angstrom})$',
        # Berry Connection: e * angström
        'dn': r'$\lvert \mb{A}_{n}(\mb{k}) \rvert \; (\si{\elementarycharge\angstrom})$',
        # Dipole Moment: e * angström
        'dnm': r'$\lvert \mb{d}_{nm}(\mb{k}) \rvert \; (\si{\elementarycharge\angstrom})$',
        # Projected Berry Connection: e * angström
        'ephi_dot_dn': r'$\lvert \hat{e}_\phi \cdot \mb{A}_{n}(\mb{k}) \rvert \; (\si{\elementarycharge\angstrom})$',
        # Projected Dipole Moment: e * angström
        'ephi_dot_dnm': r'$\lvert \hat{e}_\phi \cdot \mb{d}_{nm}(\mb{k}) \rvert \; (\si{\elementarycharge\angstrom})$',
        # Frequency: terahertz
        'f': r'$f \; (\si{\tera\hertz})$',
        # Harmonic Order
        'ff0': r'$f/f_0$',
        # Electric Field: megavolt/cm
        'E': r'$\mr{E}(t) \; (\si{\mega\volt\per\cm})$',
        # Vector Potential: megavolt * femtoseconds/cm
        'A': r'$\mr{A}(t) \; (\si{\mega\volt\per\cm\fs})$'}

def parse_cued_aliases():
	'''
	Parse the LaTeX alias file to use aliases in matplotlib routines.
	'''
	alias_file_str = cued_pwd('plotting/tex_templates/CUED_aliases.tex')
	alias_file = open(alias_file_str, 'r')

	aliases = ''

	for line in alias_file:
		if not line.startswith('%'):
			aliases += line.strip()
	
	return aliases

def init_matplotlib_config():
	packages = r'\usepackage{siunitx}' + \
             r'\usepackage{braket}' + \
             r'\usepackage{mathtools}' + \
             r'\usepackage{amssymb}' + \
             r'\usepackage[version=4]{mhchem}' + \
             r'\usepackage[super]{nth}'

	aliases = parse_cued_aliases()

	# LAYOUT
	mpl.rc('figure', figsize=(5.295, 6.0), autolayout=True)
	# LATEX
	mpl.rc('text', usetex=True)
	mpl.rc('text.latex', preamble=packages + aliases)
	# FONT
	mpl.rc('font', size=10, family='serif', serif='CMU Serif')
	# mpl.rc('mathtext', fontset='cm')
	# AXES
	mpl.rc('axes', titlesize=10, labelsize=10)
	# LEGEND
	mpl.rc('legend', fancybox=False, fontsize=10, framealpha=1, labelspacing=0.08,
	       borderaxespad=0.05)
	# PGF
	mpl.rc('pgf', texsystem='lualatex', preamble=packages + aliases)
	mpl.use('Agg')
