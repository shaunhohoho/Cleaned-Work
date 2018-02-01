# Explanation
A set of cleaned scripts for analysing the data received from pulsed measurement procedures, using a confocal microscope setup. The data SpinReadout.csv shows the fluorescence “Counts” collected within a “Detection Time (ns)” timebin of 25ns. The data shows 2 regions of high fluorescence (pulses), which last for ~3us, and reveal details of the spin state of the fluorescent sample (in this case an NV centre).

The fluorescence of the NV centre is dependent on it’s intrinsic initial spin state, and comparison of the photodynamics of the first and second pulses give a metric for the change in spin between the pulses. During the dark region between the pulses, a spin manipulation (MW) process is performed to transform the spin. 

Using information collected from [1] concerning the optical rates of the NV centre, a fluorescence profile of the NV centre during the pulses can be created, this employs a transition matrix approach to determine the photodynamics (change in population of states) over time. A variety of methods can be used (StochasticModelSolver.py) but for a large ensemble of experiments performed on a single sample such as this one, a deterministic model can be used to describe the fluorescence. (Other experiments which are performed in a single-shot fashion, as opposed to averaging over a large ensemble will require a more true stochastic description of the system).

The fluorescence profile of the pulses are then fitted using this model (PulsedMeasurementFitter.py) and the corresponding spin states associated with the fluorescence is found.

## References
[1] Manson, N. B., J. P. Harrison, and M. J. Sellars. "Nitrogen-vacancy center in diamond: Model of the electronic structure and associated dynamics." Physical Review B 74.10 (2006): 104303.
