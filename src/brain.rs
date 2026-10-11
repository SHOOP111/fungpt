use serde::Serialize;
use serde_json::{json, Value};
use std::collections::VecDeque;

const DT: f32 = 0.001;
const REGION_DEFS: [RegionDef; 7] = [
    RegionDef { name: "Visual cortex", short: "VISUAL", color: "#78a9ff", count: 48 },
    RegionDef { name: "Thalamus", short: "THALAMUS", color: "#70e1d1", count: 32 },
    RegionDef { name: "Prefrontal cortex", short: "PREFRONTAL", color: "#c4a7ff", count: 64 },
    RegionDef { name: "Hippocampus", short: "HIPPOCAMPUS", color: "#f5bd6d", count: 40 },
    RegionDef { name: "Amygdala", short: "AMYGDALA", color: "#ff8fa3", count: 28 },
    RegionDef { name: "Motor cortex", short: "MOTOR", color: "#8ed081", count: 48 },
    RegionDef { name: "Cerebellum", short: "CEREBELLUM", color: "#75c9f8", count: 32 },
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
    neurons: Vec<Neuron>,
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
    pub fn new() -> Self {
        let mut offsets = [0usize; 8];
        for i in 0..REGION_DEFS.len() {
            offsets[i + 1] = offsets[i] + REGION_DEFS[i].count;
        }

        let mut brain = Self {
            neurons: Vec::with_capacity(offsets[7]),
            synapses: Vec::new(),
            outgoing: vec![Vec::new(); offsets[7]],
            incoming: vec![Vec::new(); offsets[7]],
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
            events: VecDeque::new(),
            memories: Vec::new(),
            recall_result: "No retrieval performed yet.".to_owned(),
            rng: 0xA076_1D64_78BD_642F,
        };

        for (region, def) in REGION_DEFS.iter().enumerate() {
            for local in 0..def.count {
                let inhibitory = local % 5 == 0;
                brain.neurons.push(Neuron {
                    region,
                    v: -65.0 + brain.rand_f32() * 3.0,
                    rest: -65.0,
                    threshold: -50.5 + brain.rand_f32() * 1.8,
                    tau: 0.014 + brain.rand_f32() * 0.012,
                    synaptic_current: 0.0,
                    external_drive: 0.0,
                    tonic_drive: if region == 0 || region == 1 { 3.05 } else { 2.6 },
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

        for pre in 0..brain.neurons.len() {
            let pre_region = brain.neurons[pre].region;
            let degree = 11 + brain.rand_usize(10);
            for _ in 0..degree {
                let target_region = brain.choose_target_region(pre_region);
                let local = brain.rand_usize(REGION_DEFS[target_region].count);
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

        brain.push_event("CORTEX LAB ONLINE · 292 model neurons initialized");
        brain.push_event("Sparse recurrent connectome assembled · excitatory / inhibitory balance enabled");
        brain.push_event("Spike-timing-dependent plasticity armed");
        brain
    }

    fn rand_u64(&mut self) -> u64 {
        // Small deterministic xorshift generator: reproducible runs, no runtime RNG dependency.
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
        if upper == 0 { 0 } else { (self.rand_u64() as usize) % upper }
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
        if self.events.len() >= 18 {
            self.events.pop_back();
        }
        self.events.push_front(message.into());
    }

    fn find_region(&self, name: &str) -> Option<usize> {
        let needle = name.to_lowercase();
        REGION_DEFS.iter().position(|r| {
            r.name.to_lowercase() == needle
                || r.short.to_lowercase() == needle
                || r.name.to_lowercase().contains(&needle)
                || r.short.to_lowercase().contains(&needle)
        })
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
        self.push_event(if paused { "Simulation paused by operator" } else { "Simulation resumed" });
    }

    pub fn toggle_pause(&mut self) {
        self.set_paused(!self.paused);
    }

    pub fn clock_speed(&self) -> f32 {\n        self.speed\n    }\n\n    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.25, 8.0);
        self.push_event(format!("Simulation clock set to {:.2}×", self.speed));
    }

    pub fn toggle_plasticity(&mut self) {
        self.plasticity = !self.plasticity;
        self.push_event(if self.plasticity { "Synaptic plasticity enabled" } else { "Synaptic plasticity frozen" });
    }

    pub fn stimulate(&mut self, region_name: &str, intensity: f32) -> Result<(), String> {
        let region = self.find_region(region_name)
            .ok_or_else(|| format!("Unknown brain region: {region_name}"))?;
        let intensity = intensity.clamp(0.0, 40.0);
        let start = self.offsets[region];
        let end = self.offsets[region + 1];
        let count = ((end - start) / 3).max(6);
        for _ in 0..count {
            let idx = start + self.rand_usize(end - start);
            self.neurons[idx].external_drive += intensity * (0.55 + self.rand_f32() * 0.85);
        }
        self.acetylcholine = (self.acetylcholine + 0.07).min(1.0);
        self.noradrenaline = (self.noradrenaline + 0.035).min(1.0);
        self.push_event(format!("INPUT · {} stimulated at {:.1} mV drive", REGION_DEFS[region].name, intensity));
        Ok(())
    }

    pub fn reward(&mut self, magnitude: f32) {
        let magnitude = magnitude.clamp(0.05, 1.0);
        self.dopamine = (self.dopamine + magnitude * 0.48).min(1.0);
        self.reward_pulses += 1;
        for synapse in &mut self.synapses {
            if synapse.weight > 0.0 {
                synapse.weight = (synapse.weight * (1.0 + magnitude * 0.002)).min(0.85);
            }
        }
        self.push_event(format!("REWARD PREDICTION ERROR · dopamine pulse +{:.2}", magnitude * 0.48));
    }

    pub fn induce_stress(&mut self, magnitude: f32) {
        let amount = magnitude.clamp(0.05, 1.0);
        self.noradrenaline = (self.noradrenaline + amount * 0.52).min(1.0);
        self.serotonin = (self.serotonin - amount * 0.14).max(0.05);
        self.stimulate("Amygdala", 15.0).ok();
        self.push_event(format!("SALience / stress signal · arousal +{amount:.2}"));
    }

    pub fn novelty(&mut self) {
        self.acetylcholine = (self.acetylcholine + 0.36).min(1.0);
        self.noradrenaline = (self.noradrenaline + 0.18).min(1.0);
        self.stimulate("Visual cortex", 13.0).ok();
        self.push_event("NOVELTY DETECTED · attention and encoding gates increased");
    }

    pub fn associate(&mut self, cue: &str, target: &str) -> Result<(), String> {
        let cue = cue.trim();
        let target = target.trim();
        if cue.is_empty() || target.is_empty() {
            return Err("Both cue and target are required.".to_owned());
        }
        if cue.len() > 220 || target.len() > 220 {
            return Err("Keep each memory field under 220 characters.".to_owned());
        }
        let pattern = Self::embed(cue);
        let existing = self.memories.iter().position(|m| m.cue.eq_ignore_ascii_case(cue));
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
            if self.memories.len() >= 80 {
                self.memories.remove(0);
            }
            self.memories.push(trace);
        }
        self.acetylcholine = (self.acetylcholine + 0.23).min(1.0);
        self.dopamine = (self.dopamine + 0.18).min(1.0);
        self.drive_region("Hippocampus", 20.0);
        self.drive_region("Prefrontal cortex", 12.0);
        self.push_event(format!("ENCODING · associative trace stored for “{}”", Self::shorten(cue, 58)));
        let _ = pattern;
        Ok(())
    }

    pub fn recall(&mut self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            self.recall_result = "Enter a cue to search the associative memory.".to_owned();
            return self.recall_result.clone();
        }
        let query_embed = Self::embed(query);
        let query_tokens = Self::tokens(query);
        let best = self.memories.iter().enumerate().map(|(i, m)| {
            let emb = Self::embed(&m.cue);
            let cosine = query_embed.iter().zip(emb.iter()).map(|(a, b)| a * b).sum::<f32>();
            let tokens = Self::tokens(&m.cue);
            let intersection = query_tokens.iter().filter(|t| tokens.contains(t)).count() as f32;
            let union = (query_tokens.len() + tokens.len()).saturating_sub(intersection as usize).max(1) as f32;
            let lexical = intersection / union;
            let score = (cosine * 0.58 + lexical * 0.72) * m.strength;
            (i, score)
        }).max_by(|a, b| a.1.total_cmp(&b.1));

        let result = if let Some((index, score)) = best {
            if score >= 0.12 {
                let memory = &mut self.memories[index];
                memory.retrievals += 1;
                memory.strength = (memory.strength + 0.025).min(1.0);
                format!("{}  ·  match {:.0}%  ·  retrieved {}×", memory.target, score.mul_add(100.0, 0.0).clamp(0.0, 100.0), memory.retrievals)
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
        self.push_event(format!("RETRIEVAL · cue “{}” queried", Self::shorten(query, 48)));
        result
    }

    fn drive_region(&mut self, name: &str, intensity: f32) {
        if let Some(region) = self.find_region(name) {
            let start = self.offsets[region];
            let end = self.offsets[region + 1];
            for _ in 0..((end - start) / 2).max(4) {
                let index = start + self.rand_usize(end - start);
                self.neurons[index].external_drive += intensity * (0.65 + self.rand_f32() * 0.70);
            }
        }
    }

    pub fn sleep_consolidation(&mut self) {
        // A deliberately simplified replay/consolidation phase, not a physiological sleep model.
        for memory in &mut self.memories {
            memory.strength = (memory.strength + 0.055).min(1.0);
        }
        for synapse in &mut self.synapses {
            synapse.weight *= 0.9994;
            if synapse.weight.abs() < 0.025 {
                synapse.weight = synapse.weight.signum() * 0.025;
            }
        }
        self.acetylcholine = (self.acetylcholine * 0.55).max(0.05);
        self.noradrenaline = (self.noradrenaline * 0.62).max(0.04);
        self.serotonin = (self.serotonin + 0.08).min(1.0);
        self.push_event("OFFLINE REPLAY · memory traces strengthened; weak weights renormalized");
    }

    fn tokens(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 1)
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
        for x in &mut v { *x /= norm; }
        v
    }

    fn shorten(text: &str, max: usize) -> String {
        let mut out = text.chars().take(max).collect::<String>();
        if text.chars().count() > max { out.push('…'); }
        out
    }

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
        let mut fired_now = Vec::with_capacity(40);
        for i in 0..self.neurons.len() {
            let noise = (self.rand_f32() * 2.0 - 1.0) * noise_gain;
            let n = &mut self.neurons[i];
            n.fired = false;
            n.pre_trace *= (-DT / 0.020).exp();
            n.post_trace *= (-DT / 0.020).exp();
            n.synaptic_current *= (-DT / 0.006).exp();
            n.external_drive *= (-DT / 0.095).exp();
            n.adaptation *= (-DT / 0.180).exp();
            n.rate_hz *= (-DT / 0.45).exp();
            if n.refractory > 0.0 {
                n.refractory -= DT;
                n.v = n.rest - 1.0;
                continue;
            }
            let effective_threshold = n.threshold - self.acetylcholine * 0.65;
            let dv = (n.rest - n.v + n.synaptic_current + n.external_drive + n.tonic_drive + noise - n.adaptation)
                * DT / n.tau;
            n.v += dv;
            if n.v >= effective_threshold {
                n.fired = true;
                n.v = -62.5;
                n.refractory = 0.0025;
                n.adaptation += 1.6;
                n.spikes += 1;
                n.rate_hz += 4.0;
                fired_now.push(i);
            }
        }
        self.total_spikes += fired_now.len() as u64;

        for &pre in &fired_now {
            let pre_trace = self.neurons[pre].pre_trace;
            let inhibitory = self.neurons[pre].inhibitory;
            for edge in self.outgoing[pre].iter().copied() {
                let post = self.synapses[edge].post;
                if self.plasticity && !inhibitory {
                    let depression = 0.0010 * self.neurons[post].post_trace;
                    let w = self.synapses[edge].weight;
                    self.synapses[edge].weight = (w.abs() - depression).clamp(0.025, 0.85) * w.signum();
                }
                let weight = self.synapses[edge].weight;
                self.neurons[post].synaptic_current += weight * 74.0;
            }
            self.neurons[pre].pre_trace = (pre_trace + 1.0).min(6.0);
        }

        for &post in &fired_now {
            if self.plasticity {
                for edge in self.incoming[post].iter().copied() {
                    let pre = self.synapses[edge].pre;
                    if self.neurons[pre].inhibitory {
                        continue;
                    }
                    let w = self.synapses[edge].weight;
                    let potentiation = 0.00135 * self.neurons[pre].pre_trace;
                    self.synapses[edge].weight = (w.abs() + potentiation).clamp(0.025, 0.85) * w.signum();
                }
            }
            self.neurons[post].post_trace = (self.neurons[post].post_trace + 1.0).min(6.0);
        }

        if self.tick % 1000 == 0 {
            let active = self.neurons.iter().filter(|n| n.rate_hz > 1.0).count();
            self.push_event(format!("NETWORK · t={:.1}s · {active} responsive units · {} cumulative spikes",
                self.tick as f32 * DT, self.total_spikes));
        }
    }

    pub fn snapshot(&self) -> Value {
        let nodes: Vec<Value> = self.neurons.iter().enumerate().map(|(id, n)| json!({
            "id": id,
            "region": n.region,
            "v": (n.v * 100.0).round() / 100.0,
            "rate": (n.rate_hz * 100.0).round() / 100.0,
            "fired": n.fired,
            "inhibitory": n.inhibitory,
        })).collect();

        let regions: Vec<Value> = REGION_DEFS.iter().enumerate().map(|(region, def)| {
            let start = self.offsets[region];
            let slice = &self.neurons[start..self.offsets[region + 1]];
            let rate = slice.iter().map(|n| n.rate_hz).sum::<f32>() / slice.len() as f32;
            let active = slice.iter().filter(|n| n.rate_hz > 1.0).count();
            json!({
                "id": region,
                "name": def.name,
                "short": def.short,
                "color": def.color,
                "count": def.count,
                "mean_rate": (rate * 100.0).round() / 100.0,
                "active": active,
                "activity": active as f32 / slice.len() as f32,
            })
        }).collect();

        let stride = (self.synapses.len() / 1050).max(1);
        let links: Vec<Value> = self.synapses.iter().step_by(stride).map(|s| json!({
            "source": s.pre,
            "target": s.post,
            "weight": (s.weight * 1000.0).round() / 1000.0,
        })).collect();

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
                "plasticity": self.plasticity,
                "spikes": self.total_spikes,
                "firing_now": self.neurons.iter().filter(|n| n.fired).count(),
                "responsive": self.neurons.iter().filter(|n| n.rate_hz > 1.0).count(),
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
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_a_connected_multiregion_network() {
        let brain = Brain::new();
        assert_eq!(brain.neurons.len(), 292);
        assert!(brain.synapses.len() > 2_500);
        assert_eq!(brain.outgoing.len(), brain.neurons.len());
        assert_eq!(brain.incoming.len(), brain.neurons.len());
        assert!(brain.neurons.iter().any(|n| n.inhibitory));
        assert!(brain.neurons.iter().any(|n| !n.inhibitory));
    }

    #[test]
    fn targeted_stimulation_produces_activity() {
        let mut brain = Brain::new();
        brain.stimulate("Visual cortex", 35.0).unwrap();
        for _ in 0..1200 { brain.step(); }
        assert!(brain.total_spikes > 0, "the stimulated model should emit action potentials");
    }

    #[test]
    fn pause_freezes_simulated_time() {
        let mut brain = Brain::new();
        brain.set_paused(true);
        let before = brain.tick;
        for _ in 0..20 { brain.step(); }
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
        assert_eq!(state["neurons"].as_array().unwrap().len(), 292);
        assert!(state["links"].as_array().unwrap().len() > 100);
        assert_eq!(state["regions"].as_array().unwrap().len(), 7);
    }
}
