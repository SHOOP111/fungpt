# Cortex Lab model notes

This document describes what the program actually computes. The equations below are engineering abstractions intended for inspection and experimentation; they are not a claim of biological parameter calibration.

## Network at a glance

The default network uses 1,024 units across seven populations:

| Population | Units at 1× | Main model role |
| --- | ---: | --- |
| Visual cortex | 160 | Sensory input / recurrent visual population |
| Thalamus | 96 | Cross-population relay |
| Prefrontal cortex | 256 | Recurrent integration / task-associated drive |
| Hippocampus | 128 | Associative encoding and retrieval drive |
| Amygdala | 96 | Salience / stress modulation target |
| Motor cortex | 192 | Downstream motor-population activity |
| Cerebellum | 96 | Motor-related recurrent population |

Every neuron has 14–24 outgoing synapse attempts during construction; self-connections are skipped. This results in roughly 19,000 directed synapses at the baseline scale. Connection targets are sampled from region-biased distributions, creating both recurrent and inter-region edges. About one fifth of units are inhibitory by construction (local index modulo five equals zero); synapses from these units carry negative weight. This is synthetic topology, not a traced connectome.

Set BRAINLAB_SCALE to a value in the range 0.25–8.0 to scale population sizes together. At 8×, the model contains 8,192 units and roughly 150,000 edges. Memory use and integration speed rise with scale. The browser draws a sampled subset of edges so the interface remains usable on larger configurations.

## Integration step

The fixed base step is dt = 0.001 s (1 ms). Membrane potential is advanced with an explicit Euler step:

```text
V(t + dt) = V(t) + (dt / tau_m) *
  (V_rest - V(t) + I_syn + I_external + I_tonic + noise - adaptation)
```

- Resting potential is -65 mV.
- Each unit receives its own threshold (approximately -50.5 ± 0.9 mV) and membrane time constant (14–26 ms).
- Acetylcholine-like modulation lowers the firing threshold slightly.
- A threshold crossing emits a spike, resets voltage to -62.5 mV, and triggers a 2.5 ms refractory period.
- A spike raises the unit's adaptation term; that term decays with a 180 ms time constant.
- Synaptic current decays with a 6 ms time constant. When a presynaptic unit spikes, each outgoing edge adds its signed weight times 74 to its target current.
- Externally applied stimulus decays with a 95 ms time constant. Background drive and bounded pseudo-random noise produce ongoing subthreshold dynamics.

This is a point-neuron model: there are no dendritic compartments, action-potential waveforms, ion-channel gates or explicit axonal conduction delays. A “spike” is an event, not a simulated electrical waveform.

## Synaptic plasticity

Each unit has exponentially decaying pre- and postsynaptic traces with a 20 ms time constant. In simplified form, an excitatory edge's magnitude is adjusted when events arrive:

```text
presynaptic spike: |w| <- clamp(|w| - 0.00100 * post_trace, 0.025, 0.85)
postsynaptic spike: |w| <- clamp(|w| + 0.00135 * pre_trace, 0.025, 0.85)
```

The presynaptic edge sign is retained, and inhibitory-source edges do not undergo this excitatory STDP update. Simultaneous events are processed in discrete order within a time step, so this is an intentionally simple approximation of spike-timing-dependent plasticity. The UI can freeze or unfreeze the rule. A reward event increases dopamine-like state and applies a small multiplicative update to positive synaptic weights; it is not a detailed three-factor biological learning rule.

## Modulatory signals

Four scalar variables influence the simulation at a qualitative level:

- **Dopamine-like:** a reward-associated signal with slow exponential decay.
- **Acetylcholine-like:** increases around input/encoding and lowers threshold a little.
- **Noradrenaline-like:** arousal gain; it scales background noise and rises after novelty/stress.
- **Serotonin-like:** a slower tone that changes gently and drops under the stress perturbation.

These variables have no physical units, receptor subtypes, transporters or realistic biochemical kinetics. The dashboard presents normalized model state, not concentrations or medical values.

## Associative memory

Memory traces store a cue, target, strength, retrieval count and creation time. Cue similarity is computed from:

1. A deterministic 48-dimensional token-hash embedding, compared by cosine similarity.
2. Token Jaccard overlap, which rewards shared words.

The resulting score is multiplied by trace strength. A sufficiently strong match returns its target and increments retrieval count; successful retrieval nudges strength upward. The offline replay action nudges all trace strengths and gently renormalizes synaptic magnitudes. This is transparent, tiny associative lookup—not episodic memory, natural language comprehension or semantic reasoning.

## Reproducibility and runtime

Network construction uses a fixed-seed xorshift generator so identical scale configurations begin with the same network. Stimuli and subsequent random draws change the trajectory deterministically for a given sequence of actions, barring changes to runtime timing or implementation.

The Rust process holds state behind a mutex. A simulation thread advances the network, while a local HTTP server serves a JSON snapshot and accepts control actions. The dashboard samples state periodically; it does not compute neural dynamics itself.

## Interpretation

Use the model to explore cause and effect within this specific implementation: where stimuli are injected, how a coarse recurrent network responds, how the learning rule updates weights and how the toy memory responds to retrieval. Do not interpret its firing rates or modulator values as direct estimates of human neurophysiology.
