"""Unit-conversion constants shared across the pyadmd package."""

# UNIT-CONVERSION CONSTANTS (NAMD to OpenMM)
# NAMD .vel binary files store velocities in AKMA units:
AKMA_VEL_TO_NM_PS: float = 2.04548          # AKMA to nm/ps  (into OpenMM Context)
NM_PS_TO_AKMA_VEL: float = 1.0 / 2.04548    # nm/ps to AKMA  (out of OpenMM Context)
