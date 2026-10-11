# Experiment notebook

These short protocols are designed to make the stateful model easier to explore. Save observations outside the app if you need a permanent lab notebook; the current simulation state lives in memory and is reset when the process is restarted.

## 1. Perturbation and propagation

**Question:** does stimulation affect the target population and its connected populations?

1. Start with the default scale and wait for baseline metrics to settle.
2. Record responsive units, cumulative spikes and the seven regional mean rates.
3. Click Visual cortex at a medium stimulus intensity; wait roughly one model second.
4. Compare visual activity with thalamic, prefrontal and hippocampal rates.
5. Repeat with motor cortex as the input region, and then with a stronger stimulus.

**Readout:** responsive-unit count, cumulative spike delta, mean regional rates and the event stream. Recurrent paths are synthetic and stochastic, so exact spatial patterns are not biologically predictive.

## 2. Learning enabled versus frozen

**Question:** what changes when STDP is enabled?

1. Reset the experiment and keep the network scale unchanged.
2. Leave STDP enabled and repeatedly stimulate a chosen region, interleaving reward pulses.
3. Observe how the positive weights and firing pattern respond over time.
4. Reset, freeze STDP, and repeat the same sequence of actions.
5. Compare broad activity and whether the resulting trajectories diverge.

The reward action is a coarse reward-like perturbation plus a small positive-weight scaling rule. It is not a faithful biological reward prediction error or a full reinforcement-learning system.

## 3. Associative trace and retrieval

**Question:** how sensitive is the memory lookup to cue overlap?

1. Encode “favorite color” → “deep blue”.
2. Retrieve with “favorite color”, then try “color”, “favorite”, and “favorite animal”.
3. Observe the returned match score and retrieval count.
4. Encode a second association with shared words and compare which trace is returned.
5. Apply offline replay and inspect trace strengths.

This experiment is about a small deterministic lexical/hash similarity baseline. It cannot infer facts or reliably retrieve arbitrary semantic paraphrases.

## 4. Scale versus responsiveness

**Question:** how does network size affect throughput on your machine?

1. Run with BRAINLAB_SCALE=0.5, then 1, 2 and 4 (restart between values).
2. Compare startup-reported unit/edge counts, the model clock and dashboard polling responsiveness.
3. Repeat one identical stimulus per scale.
4. Note your hardware and whether the requested simulation speed is achieved.

Scaling increases the number of point neurons, not biological detail. If the dashboard becomes sluggish, lower the scale or simulation clock.

## 5. Modulatory perturbations

**Question:** how do control signals change the local dynamics?

1. Record the four normalized modulatory values.
2. Apply novelty, wait for the pulse to decay, then repeat with stress.
3. Deliver reward and inspect the dopamine-like trace.
4. Compare how the arousal-driven noise gain and acetylcholine-like threshold adjustment coincide with firing changes.

The modulators are internal dimensionless values defined by code; they are not physiological concentrations.

## Interpretation checklist

- Separate an observed result from the hypothesis you formed before the run.
- Repeat each condition; event timing and noise can change individual trajectories.
- Compare at the same model scale and approximate model time.
- Avoid translating these small-network results directly to a claim about people, cognition or disease.
