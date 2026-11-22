#!/bin/bash
# combo_e: object-aware fine-tunes (ft_g, ft_i) and the TrustMark specialist (ft_h) added to the candidates.
R=/srv/lesrooters/fable/runs
until [ -f $R/ft_h/ALL_DONE ]; do sleep 5; done
bash /srv/lesrooters/fable/code/redo_combo.sh combo_e 0 ft_b,ft_c,ft_d,ft_e,ft_g,ft_h,ft_i \
  'pixelseal=ft_g+ft_e*ft_g+ft_c*ft_e*ft_g+ft_e' \
  'trustmark=ft_h+ft_g*ft_h+ft_c*ft_e*ft_g*ft_h+ft_c*ft_e*ft_g' \
  'stegastamp=ft_d*ft_i+ft_b*ft_d*ft_i+ft_d'
