# Common Workflows

Four workflows cover most use cases. They differ in how much prior knowledge you have and how you can define your objective (and the cost of evaluation).

| Workflow | Known molecules? | Scoring function? | Typical use |
|----------|----------------|-------------------|-------------|
| [1. Sample and filter](#1-sample-from-a-prior-and-filter) | No | No | Early exploration, benchmarking |
| [2. Transfer Learning](#2-adapt-the-prior-with-transfer-learning-tl) | Yes | No | Focused generation from a chemical series |
| [3. Reinforcement Learning](#3-adapt-the-prior-with-reinforcement-learning) | No | Yes | Property-driven optimisation |
| [4. TL then RL](#4-transfer-learning-followed-by-reinforcement-learning) | Yes | Yes | Full drug design campaign |


---

## 1. Sample from a Prior and Filter


**When to use:** No known target relavance molecules, no defined scoring function. You want to explore what the prior generates and apply post-hoc filters.

**How it works:** The prior is a broad, unbiased generator trained on millions of drug-like molecules. Sampling a large number of molecules and filtering by computed properties (MW, LogP, QED, substructure) is a valid starting point when the design goal is loosely defined.

**Workflow:**
1. Run `sampling` with a large `num_smiles` (e.g. 10,000–50,000).
2. Filter the output CSV by computed properties using RDKit or a spreadsheet.
3. Cluster by scaffold and select representative compounds for synthesis or further optimisation.

**Limitations:** No active guidance toward a specific target. The hit rate for any target objective will be low.

See: [Sampling tutorial](sampling.md)

---

## 2. Adapt the Prior with Transfer Learning (TL)


**When to use:** You have a set of known molecules relevant to your target (actives, a chemical domain) but no well-defined scoring function (or that they are too expensive).

**How it works:** TL adapts the prior to your SMILES dataset. The model minimises the negative log-likelihood of reproducing the training molecules, which shifts the probability distribution of the generator toward that chemical series. After TL, sampling from the agent produces molecules that resemble the training set in terms of scaffold, substitution patterns, and (hopefully) target physico-chemical properties


**Workflow:**
1. Prepare a SMILES dataset of relevant molecules (20–500 compounds; larger is better).
2. Split into train/validation sets (80/20).
3. Run `transfer_learning` to produce a focused agent.
4. Sample from the agent and inspect output diversity and similarity to training set.

**Limitations:** It does not actively optimise toward a specific property, it only biases generation toward the training distribution which might not reflect the desired objective. 

See: [Transfer Learning tutorial](tl.md)

---

## 3. Adapt the Prior with Reinforcement Learning


**When to use:** You can define your design objective explicitly as a scoring function (physico-chemical properties, structural constraints, docking score, QSAR model).

**How it works:** RL uses the scoring function as a reward signal to iteratively update the agent. Each epoch, the agent generates a batch of molecules, each molecule is scored, and the agent weights are updated to increase the likelihood of generating high-scoring sequences. The prior is kept fixed as a regulariser — it prevents the agent from drifting into chemically implausible space.

Unlike TL, RL does not require a training set of known molecules. It explores and optimises simultaneously, guided entirely by the scoring function. This makes it powerful when the objective is well-defined but examples are scarce.

**Workflow:**
1. Define your scoring function (see [Scoring Function Design](scoring_function.md)).
2. Validate the scoring function on a test SMILES set using `scoring` run mode.
3. Run `staged_learning` starting from the prior (`agent_file = prior_file`).
4. Enable the diversity filter to prevent scaffold collapse.
5. Inspect the output CSV: score convergence, scaffold diversity, chemical quality.

**Limitations:** You need a well-defined and facilly computable scoring function. RL can require a lot of iterations to converge. Also you run into the risk of mode collapse (low diversity) or generating unrealistic molecules (if RL diverges from the prior too much).

See: [Reinforcement Learning tutorial](rl.md), [Scoring Function Design](scoring_function.md)

---

## 4. Transfer Learning Followed by Reinforcement Learning


**When to use:** You have both a set of known relevant molecules and a scoring function.  This is the most common use case in drug discovery projects. You have a target with known actives (or a chemical domain of interest) and you want to design new molecules that are similar but optimised for improved properties (potency, selectivity, etc.).

**How it works:** TL scopes the generative model down to a relevant region of chemical space first, analogous to focusing a search before starting optimisation. Starting RL from a TL agent rather than from the broad prior means:

- The agent begins in a region where valid, relevant chemistry already exists.
- RL has less space to explore, which leads to faster convergence.
- The risk of the agent converging to chemically unrealistic solutions is reduced.
- Expensive scoring components (e.g. docking) are evaluated on a more relevant set of molecules from the start, reducing wasted compute.

The TL step encodes structural knowledge (what the molecules look like), and the RL step encodes functional knowledge (what properties they should have). Together they constrain the search more effectively than either method alone.

**Workflow:**
1. Prepare a SMILES dataset of known relevant molecules and run `transfer_learning`.
2. Sample from the TL agent to confirm it generates chemistry resembling the training set.
3. Define your scoring function and validate it with a `scoring` run.
4. Run `staged_learning` with `agent_file` set to the TL checkpoint.
5. Optionally: use curriculum learning — stage 1 with a lightweight scoring function, stage 2 with a more expensive one (e.g. docking).

**Limitations:** It requires both a relevant training set and a well-defined scoring function. The TL step can bias the agent too much toward the training set (need to figure out when to stop), which might limit the novelty of solutions found by RL.

See: [Transfer Learning tutorial](tl.md), [Reinforcement Learning tutorial](rl.md)
