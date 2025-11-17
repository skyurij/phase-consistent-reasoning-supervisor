# Theoretical Basis for Phase-Consistent Reasoning

This document summarizes the formal concepts underlying the Phase-Consistent Reasoning Supervisor.  
These concepts are supported by multi-month experimental results documented in this repository.


## 1. Dialogue as a Dynamical System

A meaning state S evolves as:

S_{t+1} = F(S_t, U_t, M_t)

Attractors correspond to stable meaning regions.

## 2. Phase Mismatch

Semantic drift = Δφ_sem  
Abstraction jump = Δφ_lvl  
Meaning substitution = Δφ_sub

Total tension:

T = w1 * D + w2 * J + w3 * E 

## 3. Episodic Memory as Phase Trajectory

Each episode = (divergence, jump, error_score, tension).

Episodes form a trajectory across attractors:

A1 → Spiral → A2 → Spiral → A3 …

Equivalent to a walk on π1(T²).

## 4. Predictive Dynamics

T_{t+1} ≈ αT_t + βD_t + γJ_t + δE_t

Used for forecasting coherence breakdowns.
