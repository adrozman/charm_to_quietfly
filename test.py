resfile="pyxfoil_results/rotor_2/psi_007/r_014/SUI_rR29_266_4p2158_0p1134_99107p1202239.res"
Me = float(resfile.split("/")[-1].split("_")[-2].replace('p', '.'))
print(Me)
