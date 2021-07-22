(TeX-add-style-hook
 "CUED_summary"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("scrartcl" "11pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "colorlinks=true" "allcolors=blue" "pdfborder={0 0 0}" "pdfstartview={FitV}" "breaklinks" "linktocpage") ("inputenc" "utf8") ("babel" "ngerman" "english") ("caption" "labelfont=bf") ("geometry" "textheight=25cm" "left=2.3cm" "right=2.3cm" "headheight=25pt" "includehead" "includefoot" "heightrounded" "") ("txfonts" "varg")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "Efield"
    "Afield"
    "BZ"
    "j_E_dir"
    "j_ortho"
    "j_E_dir_whole_time"
    "j_ortho_whole_time"
    "Emission_total"
    "Emission_total_hann_parzen"
    "Emission_para_ortho_full_range"
    "Emission_para_ortho"
    "scrartcl"
    "scrartcl11"
    "xcolor"
    "hyperref"
    "color"
    "amsmath"
    "amsfonts"
    "amssymb"
    "inputenc"
    "enumitem"
    "babel"
    "epsfig"
    "pstricks"
    "graphics"
    "bbm"
    "caption"
    "booktabs"
    "pgfplots"
    "geometry"
    "fancyhdr"
    "url"
    "amsthm"
    "graphicx"
    "tikzpagenodes"
    "txfonts")
   (TeX-add-symbols
    '("paper" 4)
    "bE"
    "bk"
    "bA"
    "bj"
    "bd"
    "sd"
    "eqt"
    "pt"
    "coloneqqt"
    "intbzdkpi"
    "rhonnprime"
    "rhonprimen"
    "un")
   (LaTeX-add-labels
    "e1"
    "fig:Efield"
    "e2"
    "fig:kp"
    "fig:current"
    "current"
    "currentcomp"
    "currentdecomp"
    "fig:emissiontotal"
    "fig:emission"
    "emission"
    "emissiondecomp"
    "emissiondecomp2"
    "Wilhelm2021")
   (LaTeX-add-lengths
    "figureheight"
    "figurewidth"))
 :latex)

