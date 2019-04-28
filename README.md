# FPWCpy
Python codes simulating space-based wavefront sensing control (adaptive optics) for high-contrast imaging, including

1. two coronagraph instrument models, 1) WFIRST SPLC, 2) simple Vortex
2. one wavefront controller, Electric Field Conjugation (EFC)
3. three wavefront estimators, 1) batch process estimation (BPE), 2) Kalman filter, 3) extended Kalman filter (EKF)
4. two probe (sensing) policies, 1) empirical sinc probe (with empirical or optimal amplitude), 2) optimal probe
5. one system identification algorithm, 1) linear variational learning (vl) 2) linear EM and nonlinear vl are still under construction
6. other helper functions include Jacobian computation, detector noise model, etc.