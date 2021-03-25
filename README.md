# <img alt="CUED" src="/cued/branding/logo.png" height="100">
![Test workflow](https://github.com/ccmt-regensburg/CUED/actions/workflows/regression_test.yml/badge.svg)
![Test workflow for published calculations](https://github.com/ccmt-regensburg/CUED/actions/workflows/test_published_calculations.yml/badge.svg)

Package for computing the density matrix dynamics in solids exposed to ultrafast light pulses implementing the Semiconductor Bloch equations (SBE). Includes computation of  k-dependent bandstructures and dipole moments, computation of currents and emission intensity. 

<h3>Getting and running the code on Linux</h3>

To download the current version of the code, run

    git clone https://github.com/ccmt-regensburg/CUED CUED
    
Change to the directory of the code:

    cd CUED
    
Type ``pwd`` and set the outcome as pythonpath:

    export PYTHONPATH="/path/to/CUED"

Now, you can run a test, for example

    cd tests/01_Dirac_Nk1_2_Nk2_2_velocity/
    python3 runscript.py
    
The code is MPI parallel, you can run it via

    mpirun -np 2 python3 runscript.py
    
The output is written to ``time_data.dat`` (time-dependent current) and ``frequency_data.dat`` (emission spectrum). If you set ``save_latex_pdf = True``
 in ``params.py`` and if pdflatex is installed on your Linux machine, CUED will generate a graphical output pdf ``latex_pdf_files/CUED_summary.pdf`` containing plots of the bandstructure, dipoles, Brillouin zone, current, emission spectrum, ... 
