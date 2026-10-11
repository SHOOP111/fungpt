//! Cortical microcircuit model: sparse recurrent LIF network with STDP,
//! scalar neuromodulator state, and a tiny inspectable associative memory.
//!
//! Design goals: deterministic construction from a fixed seed, bounded
//! allocations in the hot loop, no panics on user-controlled input, and a
//! snapshot shape that the dashboard can render without extra computation.

use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};

const DT: f32 = 0.001;

/// Maximum number of recent event strings retained.
const EVENT_CAP: usize = 18;
/// Maximum number of associative traces retained (oldest-first eviction).
const MEMORY_CAP: usize = 80;
/// Maximum accepted characters per cue / target field.
const MEMORY_FIELD_MAX_CHARS: usize = 220;
/// Upper bound for external drive after repeated stimulation.
const DRIVE_MAX: f32 = 60.0;
/// Upper bound for a single stimulus intensity request.
const STIMULUS_MAX: f32 = 40.0;
/// Synaptic current kick per unit weight when a presynaptic spike arrives.
const SYNAPTIC_GAIN: f32 = 74.0;
/// STDP weight magnitude bounds.
const WEIGHT_MIN: f32 = 0.025;
const WEIGHT_MAX: f32 = 0.85;
/// How many sampled links the snapshot exposes at most (keeps large scales usable).
const LINK_SAMPLE_TARGET: usize = 1050;

const REGION_DEFS: [RegionDef; 7] = [
    RegionDef {
        name: "Visual cortex",
        short: "VISUAL",
        color: "#78a9ff",
        count: 160,
    },
    RegionDef {
        name: "Thalamus",
        short: "THALAMUS",
        color: "#70e1d1",
        count: 96,
    },
    RegionDef {
        name: "Prefrontal cortex",
        short: "PREFRONTAL",
        color: "#c4a7ff",
        count: 256,
    },
    RegionDef {
        name: "Hippocampus",
        short: "HIPPOCAMPUS",
        color: "#f5bd6d",
        count: 128,
    },
    RegionDef {
        name: "Amygdala",
        short: "AMYGDALA",
        color: "#ff8fa3",
        count: 96,
    },
    RegionDef {
        name: "Motor cortex",
        short: "MOTOR",
        color: "#8ed081",
        count: 192,
    },
    RegionDef {
        name: "Cerebellum",
        short: "CEREBELLUM",
        color: "#75c9f8",
        count: 96,
    },
];

#[derive(Clone, Copy)]
struct RegionDef {
    name: &'static str,
    short: &'static str,
    color: &'static str,
    count: usize,
}

#[derive(Clone)]
struct Neuron {
    region: usize,
    v: f32,
    rest: f32,
    threshold: f32,
    tau: f32,
    synaptic_current: f32,
    external_drive: f32,
    tonic_drive: f32,
    refractory: f32,
    adaptation: f32,
    pre_trace: f32,
    post_trace: f32,
    rate_hz: f32,
    spikes: u64,
    fired: bool,
    inhibitory: bool,
}

#[derive(Clone)]
struct Synapse {
    pre: usize,
    post: usize,
    weight: f32,
}

#[derive(Clone, Serialize)]
struct MemoryTrace {
    cue: String,
    target: String,
    strength: f32,
    retrievals: u64,
    created_at_s: f32,
}

pub struct Brain {
    scale: f32,
    region_counts: [usize; 7],
    neurons: Vec<Neuron>,
    // Reused every step so the hot loop never allocates.
    spike_buffer: Vec<usize>,
    synapses: Vec<Synapse>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    offsets: [usize; 8],
    tick: u64,
    paused: bool,
    speed: f32,
    plasticity: bool,
    dopamine: f32,
    acetylcholine: f32,
    noradrenaline: f32,
    serotonin: f32,
    reward_pulses: u64,
    total_spikes: u64,
    events: VecDeque<String>,
    memories: Vec<MemoryTrace>,
    recall_result: String,
    rng: u64,
}

impl Brain {
    /// Default 1x network.
    pub fn new() -> Self {
        Self::with_scale(1.0)
    }

    /// Build a network scaled around the 1024-neuron baseline.
    ///
    /// Non-finite scales fall back to 1x; finite values are clamped to
    /// 0.25..=8.0 so a hostile `BRAINLAB_SCALE` cannot OOM the process.
    pub fn with_scale(requested_scale: f32) -> Self {
        let scale = if requested_scale.is_finite() {
            requested_scale.clamp(0.25, 8.0)
        } else {
            1.0
        };
        let region_counts: [usize; 7] = std::array::from_fn(|i| {
            (REGION_DEFS[i].count as f32 * scale).round().max(1.0) as usize
        });
        let mut offsets = [0usize; 8];
        for i in 0..REGION_DEFS.len() {
            offsets[i + 1] = offsets[i] + region_counts[i];
        }
        let total = offsets[7];

        let mut brain = Self {
            scale,
            region_counts,
            neurons: Vec::with_capacity(total),
            // Worst case every neuron fires in one tick; keep a useful
            // initial capacity without reserving `total` up front.
            spike_buffer: Vec::with_capacity((total / 8).clamp(64, 2048)),
            synapses: Vec::new(),
            outgoing: vec![Vec::new(); total],
            incoming: vec![Vec::new(); total],
            offsets,
            tick: 0,
            paused: false,
            speed: 1.0,
            plasticity: true,
            dopamine: 0.20,
            acetylcholine: 0.34,
            noradrenaline: 0.18,
            serotonin: 0.56,
            reward_pulses: 0,
            total_spikes: 0,
            events: VecDeque::with_capacity(EVENT_CAP),
            memories: Vec::new(),
            recall_result: "No retrieval performed yet.".to_owned(),
            rng: 0xA076_1D64_78BD_642F,
        };

        for (region, _def) in REGION_DEFS.iter().enumerate() {
            for local in 0..brain.region_counts[region] {
                let inhibitory = local % 5 == 0;
                let v = -65.0 + brain.rand_f32() * 3.0;
                let threshold = -50.5 + brain.rand_f32() * 1.8;
                let tau = 0.014 + brain.rand_f32() * 0.012;
                brain.neurons.push(Neuron {
                    region,
                    v,
                    rest: -65.0,
                    threshold,
                    tau,
                    synaptic_current: 0.0,
                    external_drive: 0.0,
                    tonic_drive: if region == 0 || region == 1 {
                        3.05
                    } else {
                        2.6
                    },
                    refractory: 0.0,
                    adaptation: 0.0,
                    pre_trace: 0.0,
                    post_trace: 0.0,
                    rate_hz: 0.0,
                    spikes: 0,
                    fired: false,
                    inhibitory,
                });
            }
        }

        // Reserve roughly the expected edge count up front so construction
        // does not repeatedly reallocate the edge list.
        brain.synapses.reserve(total * 19);
        for pre in 0..brain.neurons.len() {
            let pre_region = brain.neurons[pre].region;
            let degree = 14 + brain.rand_usize(11);
            for _ in 0..degree {
                let target_region = brain.choose_target_region(pre_region);
                let region_size = brain.region_counts[target_region];
                if region_size == 0 {
                    continue;
                }
                let local = brain.rand_usize(region_size);
                let post = brain.offsets[target_region] + local;
                if post == pre {
                    continue;
                }
                let magnitude = 0.10 + brain.rand_f32() * 0.25;
                let weight = if brain.neurons[pre].inhibitory {
                    -magnitude
                } else {
                    magnitude
                };
                let edge = brain.synapses.len();
                brain.synapses.push(Synapse { pre, post, weight });
                brain.outgoing[pre].push(edge);
                brain.incoming[post].push(edge);
            }
        }

        let neuron_count = brain.neurons.len();
        brain.push_event(format!(
            "CORTEX LAB ONLINE · {neuron_count} model neurons initialized"
        ));
        brain.push_event(
            "Sparse recurrent connectome assembled · excitatory / inhibitory balance enabled",
        );
        brain.push_event("Spike-timing-dependent plasticity armed");
        brain
    }

    fn rand_u64(&mut self) -> u64 {
        // Small deterministic xorshift generator: reproducible runs, no runtime RNG dependency.
        // Seed is fixed non-zero; the state can never get stuck at zero from here.
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn rand_f32(&mut self) -> f32 {
        (self.rand_u64() as u32) as f32 / u32::MAX as f32
    }

    fn rand_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.rand_u64() as usize) % upper
        }
    }

    fn choose_target_region(&mut self, from: usize) -> usize {
        let choices: &[usize] = match from {
            0 => &[0, 0, 1, 1, 2, 3],
            1 => &[0, 1, 2, 2, 3, 4, 5],
            2 => &[0, 1, 2, 2, 2, 3, 3, 4, 5],
            3 => &[1, 2, 2, 3, 3, 3, 4],
            4 => &[1, 2, 3, 4, 4, 5],
            5 => &[1, 2, 5, 5, 5, 6, 6],
            _ => &[1, 5, 5, 6, 6, 6],
        };
        choices[self.rand_usize(choices.len())]
    }

    fn push_event(&mut self, message: impl Into<String>) {
        if self.events.len() >= EVENT_CAP {
            self.events.pop_back();
        }
        self.events.push_front(message.into());
    }

    fn find_region(&self, name: &str) -> Option<usize> {
        let needle = name.trim();
        if needle.is_empty() {
            return None;
        }

        // Prefer a full name or abbreviation. Partial names are accepted only when unique,
        // so an ambiguous query such as "cortex" cannot silently target the visual region.
        if let Some(index) = REGION_DEFS.iter().position(|region| {
            region.name.eq_ignore_ascii_case(needle) || region.short.eq_ignore_ascii_case(needle)
        }) {
            return Some(index);
        }

        let lowered = needle.to_lowercase();
        let mut matched: Option<usize> = None;
        for (index, region) in REGION_DEFS.iter().enumerate() {
            if region.name.to_lowercase().contains(&lowered)
                || region.short.to_lowercase().contains(&lowered)
            {
                if matched.is_some() {
                    return None; // ambiguous
                }
                matched = Some(index);
            }
        }
        matched
    }

    /// Pause or resume the simulation clock.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        self.push_event(if paused {
            "Simulation paused by operator"
        } else {
            "Simulation resumed"
        });
    }

    /// Flip the paused flag.
    pub fn toggle_pause(&mut self) {
        self.set_paused(!self.paused);
    }

    /// Current paused state.
    #[allow(dead_code)]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Current clock multiplier.
    pub fn clock_speed(&self) -> f32 {
        self.speed
    }

    /// Configured network scale (0.25..=8.0).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Number of neuron units.
    #[allow(dead_code)]
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Number of directed synapses.
    #[allow(dead_code)]
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Whether excitatory STDP updates are applied.
    #[allow(dead_code)]
    pub fn is_plastic(&self) -> bool {
        self.plasticity
    }

    /// Set the clock multiplier; non-finite values fall back to 1x.
    pub fn set_speed(&mut self, speed: f32) {
        // Keep the simulation scheduler safe even when this API is reused outside HTTP/JSON.
        self.speed = if speed.is_finite() {
            speed.clamp(0.25, 8.0)
        } else {
            1.0
        };
        let speed = self.speed;
        self.push_event(format!("Simulation clock set to {speed:.2}×"));
    }

    /// Explicitly enable or disable plasticity (prefer over toggle for UI sync).
    pub fn set_plasticity(&mut self, enabled: bool) {
        if self.plasticity != enabled {
            self.toggle_plasticity();
        }
    }

    /// Flip excitatory STDP on/off.
    pub fn toggle_plasticity(&mut self) {
        self.plasticity = !self.plasticity;
        self.push_event(if self.plasticity {
            "Synaptic plasticity enabled"
        } else {
            "Synaptic plasticity frozen"
        });
    }

    /// Inject depolarizing drive into a random subset of a region.
    pub fn stimulate(&mut self, region_name: &str, intensity: f32) -> Result<(), String> {
        if !intensity.is_finite() {
            return Err("Stimulus intensity must be a finite number.".to_owned());
        }
        let region = self
            .find_region(region_name)
            .ok_or_else(|| format!("Unknown or ambiguous brain region: {region_name}"))?;
        let intensity = intensity.clamp(0.0, STIMULUS_MAX);
        let start = self.offsets[region];
        let end = self.offsets[region + 1];
        if end <= start {
            return Err(format!(
                "Region {} has no neurons.",
                REGION_DEFS[region].name
            ));
        }
        let count = ((end - start) / 3).max(6);
        for _ in 0..count {
            let idx = start + self.rand_usize(end - start);
            self.neurons[idx].external_drive = (self.neurons[idx].external_drive
                + intensity * (0.55 + self.rand_f32() * 0.85))
                .clamp(0.0, DRIVE_MAX);
        }
        self.acetylcholine = (self.acetylcholine + 0.07).min(1.0);
        self.noradrenaline = (self.noradrenaline + 0.035).min(1.0);
        let label = REGION_DEFS[region].name;
        self.push_event(format!(
            "INPUT · {label} stimulated at {intensity:.1} mV drive"
        ));
        Ok(())
    }

    /// Dopamine-like pulse plus a small multiplicative bump to excitatory weights.
    pub fn reward(&mut self, magnitude: f32) {
        if !magnitude.is_finite() {
            self.push_event("REWARD REJECTED · magnitude must be finite");
            return;
        }
        let magnitude = magnitude.clamp(0.05, 1.0);
        self.dopamine = (self.dopamine + magnitude * 0.48).min(1.0);
        self.reward_pulses += 1;
        let factor = 1.0 + magnitude * 0.002;
        for synapse in &mut self.synapses {
            if synapse.weight > 0.0 {
                synapse.weight = (synapse.weight * factor).min(WEIGHT_MAX);
            }
        }
        self.push_event(format!(
            "REWARD PREDICTION ERROR · dopamine pulse +{:.2}",
            magnitude * 0.48
        ));
    }

    /// Arousal/salience perturbation aimed at the amygdala population.
    pub fn induce_stress(&mut self, magnitude: f32) {
        if !magnitude.is_finite() {
            self.push_event("STRESS PERTURBATION REJECTED · magnitude must be finite");
            return;
        }
        let amount = magnitude.clamp(0.05, 1.0);
        self.noradrenaline = (self.noradrenaline + amount * 0.52).min(1.0);
        self.serotonin = (self.serotonin - amount * 0.14).max(0.05);
        self.stimulate("Amygdala", 15.0).ok();
        self.push_event(format!("SALience / stress signal · arousal +{amount:.2}"));
    }

    /// Attention/encoding perturbation aimed at the visual population.
    pub fn novelty(&mut self) {
        self.acetylcholine = (self.acetylcholine + 0.36).min(1.0);
        self.noradrenaline = (self.noradrenaline + 0.18).min(1.0);
        self.stimulate("Visual cortex", 13.0).ok();
        self.push_event("NOVELTY DETECTED · attention and encoding gates increased");
    }

    /// Store (or overwrite, case-insensitively) a cue → target trace.
    pub fn associate(&mut self, cue: &str, target: &str) -> Result<(), String> {
        let cue = cue.trim();
        let target = target.trim();
        if cue.is_empty() || target.is_empty() {
            return Err("Both cue and target are required.".to_owned());
        }
        if cue.chars().count() > MEMORY_FIELD_MAX_CHARS
            || target.chars().count() > MEMORY_FIELD_MAX_CHARS
        {
            return Err(format!(
                "Keep each memory field under {MEMORY_FIELD_MAX_CHARS} characters."
            ));
        }
        let existing = self
            .memories
            .iter()
            .position(|m| m.cue.eq_ignore_ascii_case(cue));
        let trace = MemoryTrace {
            cue: cue.to_owned(),
            target: target.to_owned(),
            strength: (0.58 + self.dopamine * 0.25).min(0.92),
            retrievals: 0,
            created_at_s: self.tick as f32 * DT,
        };
        if let Some(index) = existing {
            self.memories[index] = trace;
        } else {
            if self.memories.len() >= MEMORY_CAP {
                self.memories.remove(0);
            }
            self.memories.push(trace);
        }
        self.acetylcholine = (self.acetylcholine + 0.23).min(1.0);
        self.dopamine = (self.dopamine + 0.18).min(1.0);
        self.drive_region("Hippocampus", 20.0);
        self.drive_region("Prefrontal cortex", 12.0);
        self.push_event(format!(
            "ENCODING · associative trace stored for “{}”",
            Self::shorten(cue, 58)
        ));
        Ok(())
    }

    /// Remove a single trace by cue (case-insensitive). Returns true when removed.
    pub fn forget_memory(&mut self, cue: &str) -> bool {
        let cue = cue.trim();
        if cue.is_empty() {
            return false;
        }
        if let Some(index) = self
            .memories
            .iter()
            .position(|m| m.cue.eq_ignore_ascii_case(cue))
        {
            let removed = self.memories.remove(index);
            self.push_event(format!(
                "FORGETTING · trace “{}” removed",
                Self::shorten(&removed.cue, 58)
            ));
            true
        } else {
            false
        }
    }

    /// Remove every stored trace.
    pub fn clear_memories(&mut self) {
        let count = self.memories.len();
        self.memories.clear();
        self.recall_result = "Memory store cleared.".to_owned();
        self.push_event(format!("FORGETTING · {count} associative traces cleared"));
    }

    /// Query the store with lexical + hash-embedding similarity.
    pub fn recall(&mut self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            self.recall_result = "Enter a cue to search the associative memory.".to_owned();
            return self.recall_result.clone();
        }
        let query_embed = Self::embed(query);
        let query_tokens = Self::tokens(query);
        let best = self
            .memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let emb = Self::embed(&m.cue);
                let cosine = query_embed
                    .iter()
                    .zip(emb.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                let tokens = Self::tokens(&m.cue);
                let intersection =
                    query_tokens.iter().filter(|t| tokens.contains(t)).count() as f32;
                let union = (query_tokens.len() + tokens.len())
                    .saturating_sub(intersection as usize)
                    .max(1) as f32;
                let lexical = intersection / union;
                let score = (cosine * 0.58 + lexical * 0.72) * m.strength;
                (i, score)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));

        let result = if let Some((index, score)) = best {
            if score >= 0.12 {
                let memory = &mut self.memories[index];
                memory.retrievals += 1;
                memory.strength = (memory.strength + 0.025).min(1.0);
                let pct = score.mul_add(100.0, 0.0).clamp(0.0, 100.0);
                format!(
                    "{}  ·  match {pct:.0}%  ·  retrieved {}×",
                    memory.target, memory.retrievals
                )
            } else {
                "No strong association found. Teach the model this cue first.".to_owned()
            }
        } else {
            "Memory store is empty. Teach the model a cue → target association first.".to_owned()
        };
        self.recall_result = result.clone();
        self.drive_region("Hippocampus", 15.0);
        self.drive_region("Prefrontal cortex", 9.0);
        self.acetylcholine = (self.acetylcholine + 0.10).min(1.0);
        self.push_event(format!(
            "RETRIEVAL · cue “{}” queried",
            Self::shorten(query, 48)
        ));
        result
    }

    fn drive_region(&mut self, name: &str, intensity: f32) {
        if let Some(region) = self.find_region(name) {
            let start = self.offsets[region];
            let end = self.offsets[region + 1];
            if end <= start {
                return;
            }
            for _ in 0..((end - start) / 2).max(4) {
                let index = start + self.rand_usize(end - start);
                self.neurons[index].external_drive = (self.neurons[index].external_drive
                    + intensity * (0.65 + self.rand_f32() * 0.70))
                    .clamp(0.0, DRIVE_MAX);
            }
        }
    }

    /// Simplified offline replay: strengthen traces, renormalize weak weights.
    pub fn sleep_consolidation(&mut self) {
        // A deliberately simplified replay/consolidation phase, not a physiological sleep model.
        for memory in &mut self.memories {
            memory.strength = (memory.strength + 0.055).min(1.0);
        }
        for synapse in &mut self.synapses {
            synapse.weight *= 0.9994;
            if synapse.weight.abs() < WEIGHT_MIN {
                // `signum()` maps +0.0 to +1.0, so a dead zero edge still recovers.
                synapse.weight = synapse.weight.signum() * WEIGHT_MIN;
            }
        }
        self.acetylcholine = (self.acetylcholine * 0.55).max(0.05);
        self.noradrenaline = (self.noradrenaline * 0.62).max(0.04);
        self.serotonin = (self.serotonin + 0.08).min(1.0);
        self.push_event("OFFLINE REPLAY · memory traces strengthened; weak weights renormalized");
    }

    fn tokens(text: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|token| token.chars().count() > 1)
            .filter(|token| seen.insert((*token).to_owned()))
            .map(str::to_owned)
            .collect()
    }

    fn embed(text: &str) -> Vec<f32> {
        const DIM: usize = 48;
        let mut v = vec![0.0; DIM];
        for token in Self::tokens(text) {
            let mut h = 0xcbf29ce484222325u64;
            for byte in token.bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            let idx = (h as usize) % DIM;
            v[idx] += if h & 1 == 0 { 1.0 } else { -1.0 };
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    fn shorten(text: &str, max: usize) -> String {
        let mut out = text.chars().take(max).collect::<String>();
        if text.chars().count() > max {
            out.push('…');
        }
        out
    }

    /// Advance the network by one 1 ms tick. No-op while paused.
    pub fn step(&mut self) {
        if self.paused {
            return;
        }
        self.tick += 1;
        let dopamine_decay = (-DT / 1.9).exp();
        let chol_decay = (-DT / 1.2).exp();
        let nor_decay = (-DT / 1.5).exp();
        let serotonin_decay = (-DT / 6.0).exp();
        self.dopamine = (self.dopamine * dopamine_decay + 0.00004).clamp(0.0, 1.0);
        self.acetylcholine = (self.acetylcholine * chol_decay + 0.000025).clamp(0.0, 1.0);
        self.noradrenaline = (self.noradrenaline * nor_decay + 0.00002).clamp(0.0, 1.0);
        self.serotonin = (self.serotonin * serotonin_decay + 0.000006).clamp(0.0, 1.0);

        let noise_gain = 0.8 + self.noradrenaline * 2.2;
        let trace_decay = (-DT / 0.020).exp();
        let current_decay = (-DT / 0.006).exp();
        let external_decay = (-DT / 0.095).exp();
        let adaptation_decay = (-DT / 0.180).exp();
        let rate_decay = (-DT / 0.45).exp();
        self.spike_buffer.clear();
        for i in 0..self.neurons.len() {
            let noise = (self.rand_f32() * 2.0 - 1.0) * noise_gain;
            let n = &mut self.neurons[i];
            n.fired = false;
            n.pre_trace *= trace_decay;
            n.post_trace *= trace_decay;
            n.synaptic_current *= current_decay;
            n.external_drive *= external_decay;
            n.adaptation *= adaptation_decay;
            n.rate_hz *= rate_decay;
            if n.refractory > 0.0 {
                n.refractory -= DT;
                n.v = n.rest - 1.0;
                continue;
            }
            let effective_threshold = n.threshold - self.acetylcholine * 0.65;
            let dv = (n.rest - n.v + n.synaptic_current + n.external_drive + n.tonic_drive + noise
                - n.adaptation)
                * DT
                / n.tau;
            n.v += dv;
            // Keep adversarial / buggy callers from injecting NaN through public setters.
            if !n.v.is_finite() {
                n.v = n.rest;
                n.synaptic_current = 0.0;
                n.external_drive = 0.0;
                n.adaptation = 0.0;
                continue;
            }
            if n.v >= effective_threshold {
                n.fired = true;
                n.v = -62.5;
                n.refractory = 0.0025;
                n.adaptation += 1.6;
                n.spikes += 1;
                n.rate_hz += 4.0;
                self.spike_buffer.push(i);
            }
        }
        self.total_spikes += self.spike_buffer.len() as u64;

        for &pre in &self.spike_buffer {
            let pre_trace = self.neurons[pre].pre_trace;
            let inhibitory = self.neurons[pre].inhibitory;
            // Copy edge list indices first: `outgoing[pre]` is never mutated here,
            // but copying avoids holding a borrow across the weight update.
            for edge in self.outgoing[pre].iter().copied() {
                let post = self.synapses[edge].post;
                if self.plasticity && !inhibitory {
                    let depression = 0.0010 * self.neurons[post].post_trace;
                    let w = self.synapses[edge].weight;
                    self.synapses[edge].weight =
                        (w.abs() - depression).clamp(WEIGHT_MIN, WEIGHT_MAX) * w.signum();
                }
                let weight = self.synapses[edge].weight;
                self.neurons[post].synaptic_current += weight * SYNAPTIC_GAIN;
            }
            self.neurons[pre].pre_trace = (pre_trace + 1.0).min(6.0);
        }

        for &post in &self.spike_buffer {
            if self.plasticity {
                for edge in self.incoming[post].iter().copied() {
                    let pre = self.synapses[edge].pre;
                    if self.neurons[pre].inhibitory {
                        continue;
                    }
                    let w = self.synapses[edge].weight;
                    let potentiation = 0.00135 * self.neurons[pre].pre_trace;
                    self.synapses[edge].weight =
                        (w.abs() + potentiation).clamp(WEIGHT_MIN, WEIGHT_MAX) * w.signum();
                }
            }
            self.neurons[post].post_trace = (self.neurons[post].post_trace + 1.0).min(6.0);
        }

        if self.tick % 1000 == 0 {
            let active = self.neurons.iter().filter(|n| n.rate_hz > 1.0).count();
            let elapsed = self.tick as f32 * DT;
            let total = self.total_spikes;
            self.push_event(format!(
                "NETWORK · t={elapsed:.1}s · {active} responsive units · {total} cumulative spikes"
            ));
        }
    }

    /// Full dashboard snapshot. Holds no locks itself; callers must synchronize.
    pub fn snapshot(&self) -> Value {
        // Build per-neuron payload once and reuse the aggregates for `meta`
        // so large networks are scanned as few times as possible.
        let mut nodes: Vec<Value> = Vec::with_capacity(self.neurons.len());
        let mut firing_now: usize = 0;
        let mut responsive: usize = 0;
        for (id, n) in self.neurons.iter().enumerate() {
            if n.fired {
                firing_now += 1;
            }
            if n.rate_hz > 1.0 {
                responsive += 1;
            }
            nodes.push(json!({
                "id": id,
                "region": n.region,
                "v": (n.v * 100.0).round() / 100.0,
                "rate": (n.rate_hz * 100.0).round() / 100.0,
                "fired": n.fired,
                "inhibitory": n.inhibitory,
            }));
        }

        let mut regions: Vec<Value> = Vec::with_capacity(REGION_DEFS.len());
        for (region, def) in REGION_DEFS.iter().enumerate() {
            let start = self.offsets[region];
            let end = self.offsets[region + 1];
            let slice = &self.neurons[start..end];
            let rate = if slice.is_empty() {
                0.0
            } else {
                slice.iter().map(|n| n.rate_hz).sum::<f32>() / slice.len() as f32
            };
            let active = slice.iter().filter(|n| n.rate_hz > 1.0).count();
            let activity = if slice.is_empty() {
                0.0
            } else {
                active as f32 / slice.len() as f32
            };
            regions.push(json!({
                "id": region,
                "name": def.name,
                "short": def.short,
                "color": def.color,
                "count": self.region_counts[region],
                "mean_rate": (rate * 100.0).round() / 100.0,
                "active": active,
                "activity": activity,
            }));
        }

        let stride = (self.synapses.len() / LINK_SAMPLE_TARGET).max(1);
        let link_estimate = (self.synapses.len() + stride - 1) / stride;
        let mut links: Vec<Value> = Vec::with_capacity(link_estimate);
        for s in self.synapses.iter().step_by(stride) {
            links.push(json!({
                "source": s.pre,
                "target": s.post,
                "weight": (s.weight * 1000.0).round() / 1000.0,
            }));
        }

        json!({
            "meta": {
                "name": "CORTEX LAB",
                "version": env!("CARGO_PKG_VERSION"),
                "neurons": self.neurons.len(),
                "synapses": self.synapses.len(),
                "tick": self.tick,
                "time_s": (self.tick as f32 * DT * 100.0).round() / 100.0,
                "paused": self.paused,
                "speed": self.speed,
                "scale": self.scale,
                "plasticity": self.plasticity,
                "spikes": self.total_spikes,
                "firing_now": firing_now,
                "responsive": responsive,
                "reward_pulses": self.reward_pulses,
            },
            "regions": regions,
            "neurons": nodes,
            "links": links,
            "neuromodulators": {
                "dopamine": self.dopamine,
                "acetylcholine": self.acetylcholine,
                "noradrenaline": self.noradrenaline,
                "serotonin": self.serotonin,
            },
            "memories": self.memories,
            "recall_result": self.recall_result,
            "events": self.events,
        })
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_matching_rejects_empty_and_ambiguous_names() {
        let brain = Brain::new();
        assert_eq!(brain.find_region(" visual cortex "), Some(0));
        assert_eq!(brain.find_region("VISUAL"), Some(0));
        assert_eq!(brain.find_region("motor cortex"), Some(5));
        assert_eq!(brain.find_region(""), None);
        assert_eq!(brain.find_region("   "), None);
        assert_eq!(brain.find_region("cortex"), None);
    }

    #[test]
    fn non_finite_inputs_do_not_poison_the_model() {
        let mut brain = Brain::new();
        assert!(brain.stimulate("Visual cortex", f32::NAN).is_err());
        assert!(brain.stimulate("cortex", 20.0).is_err());
        brain.reward(f32::NAN);
        brain.induce_stress(f32::INFINITY);
        brain.set_speed(f32::NAN);
        assert_eq!(brain.clock_speed(), 1.0);
        assert!(brain.dopamine.is_finite());
        assert!(brain.noradrenaline.is_finite());
        assert!(brain
            .neurons
            .iter()
            .all(|neuron| { neuron.v.is_finite() && neuron.external_drive.is_finite() }));
    }

    #[test]
    fn repeated_words_do_not_inflate_lexical_similarity() {
        assert_eq!(
            Brain::tokens("favorite color"),
            Brain::tokens("favorite favorite color favorite")
        );
    }

    #[test]
    fn repeated_stimulation_keeps_external_drive_bounded() {
        let mut brain = Brain::new();
        for _ in 0..100 {
            brain.stimulate("Amygdala", 40.0).unwrap();
        }
        assert!(brain
            .neurons
            .iter()
            .all(|neuron| neuron.external_drive <= 60.0));
    }

    #[test]
    fn configured_scale_changes_network_size_with_a_safe_bound() {
        let doubled = Brain::with_scale(2.0);
        assert_eq!(doubled.neurons.len(), 2048);
        let bounded = Brain::with_scale(1000.0);
        assert_eq!(bounded.neurons.len(), 8192);
        let invalid = Brain::with_scale(f32::NAN);
        assert_eq!(invalid.neurons.len(), 1024);
    }

    #[test]
    fn initializes_a_connected_multiregion_network() {
        let brain = Brain::new();
        assert_eq!(brain.neurons.len(), 1024);
        assert!(brain.synapses.len() > 14_000);
        assert_eq!(brain.outgoing.len(), brain.neurons.len());
        assert_eq!(brain.incoming.len(), brain.neurons.len());
        assert!(brain.neurons.iter().any(|n| n.inhibitory));
        assert!(brain.neurons.iter().any(|n| !n.inhibitory));
        assert!(brain
            .synapses
            .iter()
            .any(|s| { brain.neurons[s.pre].region != brain.neurons[s.post].region }));
    }

    #[test]
    fn targeted_stimulation_produces_activity() {
        let mut brain = Brain::new();
        brain.stimulate("Visual cortex", 35.0).unwrap();
        for _ in 0..1200 {
            brain.step();
        }
        assert!(
            brain.total_spikes > 0,
            "the stimulated model should emit action potentials"
        );
    }

    #[test]
    fn pause_freezes_simulated_time() {
        let mut brain = Brain::new();
        brain.set_paused(true);
        let before = brain.tick;
        for _ in 0..20 {
            brain.step();
        }
        assert_eq!(before, brain.tick);
    }

    #[test]
    fn associative_memory_can_be_taught_and_recalled() {
        let mut brain = Brain::new();
        brain.associate("favorite color", "deep blue").unwrap();
        let answer = brain.recall("favorite color");
        assert!(answer.contains("deep blue"), "unexpected recall: {answer}");
        assert_eq!(brain.memories[0].retrievals, 1);
    }

    #[test]
    fn snapshot_exposes_visualization_data() {
        let brain = Brain::new();
        let state = brain.snapshot();
        assert_eq!(state["neurons"].as_array().unwrap().len(), 1024);
        assert!(state["links"].as_array().unwrap().len() > 100);
        assert_eq!(state["regions"].as_array().unwrap().len(), 7);
    }
}
