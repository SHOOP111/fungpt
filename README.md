# Cortex Lab — brain-inspired systems workbench

Cortex Lab is a locally hosted, interactive research workbench built in Rust. Its core is a running, sparse, recurrent spiking-neural-network simulation; the browser interface is a live instrument panel over the actual model state, not a prerecorded animation.

> **Scientific scope:** this is a reduced-order brain-inspired model, not a replication of a human brain, a model of consciousness, or a clinically validated neuroscience tool. Its network size, simplified neurons, modulators, plasticity and memory are chosen to make mechanisms inspectable on an ordinary computer.

## What is implemented

- **1,024 stateful neuron units** distributed among seven named functional populations: visual cortex, thalamus, prefrontal cortex, hippocampus, amygdala, motor cortex and cerebellum.
- **Approximately 19,000 directed recurrent synapses**, with excitatory and inhibitory weights, local recurrence and cross-region pathways.
- **Leaky integrate-and-fire dynamics** with membrane voltage, threshold, refractory periods, synaptic-current decay, adaptation, noisy background drive and per-neuron activity estimates.
- **Spike-timing-dependent plasticity (STDP)** with pre/post traces, bounded weights and a live switch to freeze learning.
- **Four synthetic neuromodulator state variables** (dopamine-like reward, acetylcholine-like attention/encoding, noradrenaline-like arousal and a slow serotonin-like tone). These are explanatory abstractions, not biochemically calibrated concentration models.
- **Inspectable associative memory**: teach cue → target traces, query them with a small deterministic lexical/hash embedding, track retrievals, forget single cues or clear the store, and apply a simplified offline replay/consolidation pass.
- **Interactive connectome canvas** with activity halos, excitatory/inhibitory pathway tinting, region labels, hover/touch voltage readout, region-targeted stimulation, keyboard-accessible region controls, and a recent population activity trace.
- **Live event stream, speed control, pause/resume, reward, novelty and stress perturbations**, plus reset, explicit plasticity set, and memory management commands.
- **Local HTTP/JSON API**, deterministic built-in pseudo-random generator, no telemetry, hardened std-only HTTP layer, and a no-framework frontend served by the Rust process itself.

The associative-memory retrieval is an intentionally small, inspectable baseline—not human semantic memory. The network does not learn language, reason like a person or autonomously develop human-level cognition.

## Run it

Requirements: stable Rust and Cargo. No Node.js, browser build tool, external service or GPU is required.

```bash
git clone https://github.com/SHOOP111/fungpt.git
cd fungpt
cargo run --release
```

Open **http://127.0.0.1:8787** in a browser. Keep the Rust process running while using the dashboard; the browser UI and API are served by the same process. Stop with `Ctrl+C`.

The embedded HTTP server bounds request-line and header sizes, validates header values, rejects duplicate `Content-Length`, requires `application/json` when a control body declares a content type, and limits concurrent clients (64) with per-request deadlines. It accepts JSON request bodies only on `POST /api/control` and does not support chunked transfer encoding. Unknown methods on known routes return `405` with an `Allow` header; unknown routes return `404`. Security headers include `nosniff`, `DENY` framing, `no-referrer`, a same-origin `Content-Security-Policy`, and `no-store` caching. There is a `204` favicon route so browser favicon lookups do not pollute logs. Keep the default loopback binding for local use: there is no authentication, so setting `BRAINLAB_BIND` to a LAN-facing address exposes the experiment controls to clients that can reach that address (the server prints a warning when the bind is not loopback).

To change the bind address, set `BRAINLAB_BIND` before starting the app. By default it binds to loopback only, so it is not exposed to the local network.

### Scale the network

The default scale is 1.0 (1,024 units and about 19,000 synapses). `BRAINLAB_SCALE` scales every population from 0.25× to 8× baseline; invalid values fall back to 1× and out-of-range values are clamped. Larger networks use more CPU and memory, and may not sustain real-time speed on every machine.

Linux / macOS:
```bash
BRAINLAB_SCALE=2 cargo run --release
```

Windows PowerShell:
```powershell
$env:BRAINLAB_SCALE="2"
cargo run --release
```

That 2× configuration contains 2,048 units. The dashboard and reset action keep using the configured scale.

### Quick experiments

1. Click a region chip, region row, or neuron cluster to inject input and observe downstream activity (touch taps work; region rows are keyboard-focusable buttons).
2. Increase stimulus intensity; compare the response at different model clock speeds. Intensity, pathway, and label preferences persist in `localStorage`.
3. Set STDP explicitly, deliver rewards, and watch connection / firing patterns evolve.
4. Enter a cue and target in **Associative memory lab**, encode them, then retrieve by the cue or overlapping words. Click a stored trace to load it, Shift+Click to forget it, or use Forget / Clear all.
5. Pause, inspect voltage and population state, then resume. Try the offline replay action to apply the model's simplified consolidation pass.
6. Use stress and novelty actions to perturb modulatory state and compare responses.

Keyboard shortcuts: **Space** pauses/resumes, **R** delivers a reward pulse, and **N** applies novelty (when focus is not inside a control).

## HTTP API

The API is intentionally small and served from the same origin:

- `GET /` — dashboard
- `GET /healthz` — health check
- `GET /favicon.ico` — empty `204` (avoids 404 noise)
- `GET /api/state` — current JSON snapshot: metrics, regions, neurons, sampled links, modulators, memories and events
- `POST /api/control` — execute an experiment command (JSON only)

Example request:

```bash
curl -X POST http://127.0.0.1:8787/api/control \
  -H "Content-Type: application/json" \
  -d '{"action":"stimulate","region":"Hippocampus","intensity":24}'
```

Supported actions include `stimulate`, `reward`, `stress`, `novelty`, `pause`, `toggle_pause`, `speed`, `plasticity`, `associate`, `recall`, `forget`, `clear_memory` (alias `clear_memories`), `sleep` and `reset`. An association request uses `{"action":"associate","cue":"favorite color","target":"deep blue"}`; retrieval uses `{"action":"recall","query":"favorite color"}`; single-trace removal uses `{"action":"forget","cue":"favorite color"}`; full clear uses `{"action":"clear_memory"}`. Plasticity accepts an explicit `{"action":"plasticity","enabled":true}` (or `"value":1/0` for simple clients); when neither is supplied it toggles for backwards compatibility. Speed responses report the applied clock (`model.clock_speed()`), so non-finite inputs safely report `1.00×`.

## Architecture

```text
src/brain.rs
  Neural state, sparse connectome, simulation step, STDP,
  modulator abstractions, associative traces, JSON snapshot
        |
        | Arc<RwLock<Brain>> (write: step/control, read: snapshot)
        v
src/main.rs
  Fixed-step simulation thread + local std::net HTTP server
        |
        | same-origin JSON
        v
web/index.html
  Canvas connectome, live metrics, perturbation controls,
  modulator plots, associative memory panel, event stream
```

The implementation favors a small dependency surface: Rust standard library for concurrency and networking, Serde for command/snapshot JSON, and a plain HTML/CSS/JavaScript dashboard. `RwLock` lets concurrent dashboard tabs share snapshots without serializing reads; simulation and control take the write side. The snapshot builds per-neuron payload once and reuses firing/responsive aggregates for `meta` to keep large scales (up to 8,192 units / ~150k edges, ~1,050 sampled links) usable. The frontend caches layout by network signature, throttles polling when hidden/offline, honors `prefers-reduced-motion`, and uses fetch timeouts plus a single-in-flight guard.

## Development

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo build --release
cargo fmt --all -- --check
```

The test suite covers network initialization/connectivity, response to stimulation, paused-clock behavior, associative recall, snapshot shape, JSON command parsing, request-line validation, ambiguous region names, finite numeric inputs, bounded stimulation drive and duplicate-token handling. GitHub Actions runs formatting, Clippy, tests, and an optimized build on Linux and Windows for pushes and pull requests.

## Limitations worth keeping in mind

- The 1,024 units are still a deliberately reduced-order model; the human brain has orders of magnitude more neurons and vastly more complex cell types and connectivity.
- LIF dynamics, topology, time constants and weights are simplified; explicit axonal conduction and synaptic-delay queues are not modeled. There are no realistic morphologies, detailed ion channels, glia, endocrine loops or whole-brain anatomical constraints.
- The four modulator variables are qualitative control signals, not physical measurements. Reward-induced weight updates and replay are exploratory rules.
- The memory lookup uses token overlap plus a deterministic hash embedding. It is useful for inspecting association and retrieval mechanics, not evidence of semantic understanding.
- Faster-than-real-time speed is best-effort and depends on CPU, browser refresh and network size.
- A high activity rate in this model should not be interpreted as a medically or biologically meaningful measurement.

## License

MIT. See [LICENSE](LICENSE).
