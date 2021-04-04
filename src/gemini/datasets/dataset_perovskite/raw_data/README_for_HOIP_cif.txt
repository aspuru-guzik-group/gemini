The hybrid organic-inorganic dataset includes 1346 CIF files, each of them provides the optimized structure and the accompanied material properties calculated with first-principles calculations using VASP. The standard inputs used for this dataset are given in terms of an INCAR file as below

PREC       =   Accurate
ENCUT      = 400
EDIFF      =   1.d-4
ISMEAR     =   0
SIGMA      =   0.01
EDIFFG     =  -1.0E-02
IBRION     =   1
NSW        = 100
ISIF       =   3
# vdw-DF2
GGA        =   ML
LUSE_VDW   =  .TRUE.
Zab_vdW    =  -1.8867
AGGAC      =   0.0000