# statistical_shm
Dynamic identification and structural health monitoring of real structures using statistical time series analysis. Project as part of the 2020 taught course [Dynamic identification and SHM] at the University of Patras, Greece.

## The systems

The systems analyzed are a real truss in the lab and the suspension system of a real train. The data from the truss (sampled at 256 Hz) are included in the `DATA_1047304.mat` and include two pairs of excitation & respose signals of the structure in healthy condition (variable `Signals`) and four pairs of excitation & respose signals from the structure in faulty condition (varialble `Faults`) (two different faulty conditions, two pairs of signals for each of them). Finally four pairs of excitation & response signals were taken from the structure in unknown condition.

The data from the train suspension system (sampled at 980 Hz) include response-only signals of the suspension in 3 different healthy condition (variable `Baseline_signals`, 10 colums for each condition). Response-only signals at an unknown condition healthy or damaged are provided (variable `Inspection_Unknown_Signals`).

The goals of the project are:
* Dynamic identification of the structures using the sampled data.
* Identify the unknown condition of the truss (healthy or faulty, and which kind of fault).
* Dynamic identification of the suspension system.
* Identify the condition of the train suspension (healthy or damaged and which of the known healthy conditions).


<p float="left">
  <img src="https://user-images.githubusercontent.com/75118133/159879734-9499787e-e542-4cdf-a819-8d0bde150293.png" width="250" />
  <img width="10" />
  <img src="https://user-images.githubusercontent.com/75118133/159879791-7119f632-c261-45e9-8ed1-4442d611095e.png" width="300" /> 
</p>
(photos from [SMSA lab])

## Dynamic identification
Dynamic identification of the systems is **solely data-based**. The main stages are:

:heavy_check_mark: Preliminary signal analysis (normalize signals, check signals for normal probability distribution, divide signals in estimation and validation part, spectrogram for realizing the content of the signals in the frequency domain). Identification will be done using the estimation part and the validaiton part will be used for validaiton check

:heavy_check_mark: Non parametric dynamic identification of the truss in healthy structure (Welch-based)
  * Welch-based spectrum, Power Spectral Density (PSD) for autocovariance and cross-covariance
  * The excitations are checked for the white noise assumption (we have tried to excite the structure in a large frequency bandwidth, thus with white noise)
  * Frequency response function (FRF), coherence function

:heavy_check_mark: Parametric dynamic identification of the truss in healthy structure (models of AutoRegressive Moving Average with eXogenous excitation [ARMAX] family)
  * Model the truss with ARMAX model based on BIC and RSS/SSS citerion, frequency stabilization plot.
  * Compare the FRF of ARMAX models of different order and its convergence wrt the order and select the order of the ARMAX model (ARMAX(70,70,70) was selected).
  * Check the validity of the model through the residuals' ACF (for a good model they should be close to white noise), one step-ahead prediction for the validation part of the signal (since this was not used for modeling) and CCF between excitation and residuals.
  * Compare the parametric and non-parametric identification.
  * Find roots and zeros of the model (pole-zero map) and the main frequency modes (natural frequency and damping).

:heavy_check_mark: Parametric identificaiton of the healthy truss using the subspase method (state-space models) with a procedure similar to ARMAX. The method is very time consuming.

:heavy_check_mark: Compare all the dynamic identificaiton models of the healthy truss (Welch-based vs ARMAX vs State-space).

<img src="https://user-images.githubusercontent.com/75118133/159889357-7f62a45d-ff8d-4f21-b500-0b57da328d27.png" width="350" />

The same procedure can be followed for the suspension system but AR model is selected based on response-only signals under each healthy condition.

## Structural health monitoring

Based on the above dynamic identification, the monitoring of the structural health of the structured can be done using parametric or non-parametric methods.

:heavy_check_mark: Identification of the unknown condition (healthy or faulty) of the truss.
* Non parametric SHM using the unknown condition signals (PSD, FRF-based methods).
* Parametric methods (model parameter-based, residual variance-based, residual uncorrelatedness-based).
* Final decision regarding the unknown condition.

:heavy_check_mark: Identification of the type of fault for the decided faulty signals using the same above methods.

:heavy_check_mark: SHM of the suspension system under varying conditions using the [multiple-models methods].

  <img src="https://user-images.githubusercontent.com/75118133/159891843-17910a40-caf9-4b96-8739-840f5364faa3.png" width="250" />
  (photo from [multiple-models methods] paper.)


<img src="https://user-images.githubusercontent.com/75118133/159892321-d054d1f7-9fba-4ff5-b55b-8a50e4d48c21.png" width="400" />

## Running the code

* Run matlab file

## Contributors and acknowledgments
This project was conducted at the Department of Mechanical Engineering and Aeronautics at the University of Patras, Greece in 2020 by Ioannis Dadiotis as part of the [Dynamic identification and SHM] course by Prof. S. Fassois and Ass. professor J. Sakellariou of the [SMSA lab]. The lab is focused on high-performance and intelligent Stochastical MEchanical, Aeronautical, Industrial and related systems that operate under uncertainty.

<img src="https://user-images.githubusercontent.com/75118133/159381029-ff271c1e-f995-42a1-a11a-2c50890c7e5e.png" width="150" />

[SMSA lab]: https://sites.google.com/g.upatras.gr/smsa-lab/home
[Dynamic identification and SHM]: https://www.mead.upatras.gr/en/courses/domiki-akeraiotita-kataskeuwn/
[multiple-models methods]: https://ui.adsabs.harvard.edu/abs/2018MSSP..111..149V/abstract
