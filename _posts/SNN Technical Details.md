## Leaky Integrate-and-Fire Model

**Analogy:**
Imagine a bucket with a small hole at the bottom. You pour water (representing input signals) into the bucket. If you pour enough water quickly enough, the bucket overflows (the neuron spikes). Over time, water leaks out through the hole, representing the loss of input signal over time.

**Technical Breakdown:**
- **Membrane Potential (V):** The neuron's state is described by its membrane potential $V(t)$. When a neuron receives input, this potential increases.
- **Leakage:** Over time, the potential decreases due to leakage, akin to water slowly leaking out of the bucket. This is modeled by a term that decays the potential exponentially.
- **Threshold:** When the membrane potential reaches a certain threshold $V_{th}$, the neuron fires (spikes), and the potential is reset to a lower value (often zero or a resting potential).
- **Dynamics Equation:** The dynamics of the LIF neuron can be described by the differential equation:
$$
\tau_m \frac{dV(t)}{dt} = -V(t) + R \cdot I(t)
$$
  where:
  -  $\tau_m$ is the membrane time constant.
  -  $R$ is the membrane resistance.
  -  $I(t)$ is the input current.

**Spiking:** When $V(t)$ reaches $V_{th}$ , the neuron emits a spike, and $V(t)$ is reset.
## Hodgkin-Huxley model 

## Spike-Timing-Dependent Plasticity (STDP)

