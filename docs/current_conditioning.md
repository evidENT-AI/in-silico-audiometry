# Prior Conditioning Service - Current State (January 2025)

This document describes the current state of prior conditioning for both the in-situ clinical system and the in-silico simulation codebase. Includes clinical expert review (Roulla Katiri, January 2025).

## Overview

Prior conditioning generates informed Bayesian priors for audiometry based on patient data. The system implements a three-level hierarchical structure:

1. **Level 1**: Age-sex stratification (from NHANES empirical data)
2. **Level 2**: Risk factor adjustments (from NHANES + literature)
3. **Level 3**: Tympanometric conditioning

## NHANES-Based Empirical Priors

The in-silico simulation codebase now includes NHANES data acquisition and prior construction:

- **Location**: `audiometry_ai/data/` and `audiometry_ai/priors/`
- **Script**: `scripts/build_nhanes_priors.py`
- **Data**: NHANES audiometry (AUX), demographics (DEMO), questionnaires (DIQ, MCQ, BPQ)

### To build priors:
```bash
python scripts/build_nhanes_priors.py --download --years 2015-2016 2017-2018
```

---

## Clinical Expert Notes (Roulla Katiri, January 2025)

### ⚠️ IMPORTANT CORRECTIONS

1. **Type Ad Tympanometry**: Do NOT use as a prior for hearing loss. Hypermobile tympanic membranes can occur for many reasons unrelated to hearing loss. Implementation now only increases uncertainty without shifting mean.

2. **Cardiovascular Risk**: Affects **LOW frequencies**, not high frequencies as sometimes stated. The apex of the cochlea (which hosts low-frequency hair cells) is poorly perfused when cardiovascular function is compromised.

3. **Meniere's Disease**: Usually shows **LOW frequency** hearing loss. Also characterized by fluctuating thresholds requiring increased uncertainty.
   - References: ASHA JSLHR 42(4):829, PMC12250289

4. **Ototoxic Medications**: Causes **HIGH frequency** loss (4-6-8 kHz onwards), often with ski-slope pattern. This is anecdotal but consistent clinically.

5. **Head Trauma**: Can cause cognitive impairment/processing difficulties in addition to hearing effects.

6. **Vertigo/Balance**:
   - If due to Meniere's: LOW frequency effects only
   - If due to vestibular schwannoma: HIGH frequency ski-slope loss
   - Unknown etiology: increase uncertainty only

---

## Parameters ACTIVELY Used for Prior Conditioning

### 1. Age (Presbycusis Model) - Level 1
- **Weight**: 0.4
- **Effect**: Frequency-dependent threshold elevation starting at age 30
  - 250 Hz: 0.05 factor
  - 500 Hz: 0.10 factor
  - 1000 Hz: 0.15 factor
  - 2000 Hz: 0.25 factor
  - 4000 Hz: 0.50 factor
  - 8000 Hz: 0.80 factor
- **Source**: NHANES empirical data
- **Status**: ✅ Fully implemented in both systems

### 2. Sex/Gender - Level 1
- **Effect**: Males ~5-7 dB worse at high frequencies
  - 250 Hz: 0 dB
  - 500 Hz: 1 dB
  - 1000 Hz: 2 dB
  - 2000 Hz: 3 dB
  - 4000 Hz: 5 dB
  - 8000 Hz: 7 dB
- **Source**: NHANES (Hoffman et al. 2017)
- **Status**: ✅ Implemented in simulation; placeholder in-situ

### 3. Previous Audiogram
- **Weight**: 0.8 (strongest prior)
- **Effect**: Gaussian prior centered on previous thresholds with ±10 dB SD
- **Status**: ✅ Fully implemented in-situ; N/A in simulation

### 4. HHIA Total Score
- **Effect**: Shifts mean based on self-reported handicap
  - ≤16 (no handicap): 15 dB mean, 0.3 confidence
  - 17-42 (mild-moderate): 35 dB mean, 0.5 confidence
  - >42 (significant): 55 dB mean, 0.6 confidence
- **Status**: ✅ Fully implemented in-situ

### 5. Noise Exposure History - Level 2
- **Effect**: 4 kHz notch pattern
  - 250 Hz: 0 dB
  - 500 Hz: 0 dB
  - 1000 Hz: 2 dB
  - 2000 Hz: 5 dB
  - 3000 Hz: 15 dB (notch begins)
  - 4000 Hz: 25 dB (maximum notch)
  - 6000 Hz: 15 dB (recovery)
  - 8000 Hz: 10 dB (partial recovery)
- **Source**: NHANES + NIHL literature
- **Status**: ✅ Fully implemented in both systems

### 6. Tympanometry - Level 3
- **Effect**: Ear-specific adjustments based on type
  - **Type A**: No adjustment (normal middle ear)
  - **Type As**: +10-15 dB low-freq (stiffness/otosclerosis)
  - **Type Ad**: ⚠️ NO MEAN SHIFT - only increase uncertainty (clinical note)
  - **Type B**: +25 dB across frequencies (effusion/perforation)
  - **Type C**: +8-15 dB low-freq (ETD)
- **Status**: ✅ Fully implemented (CORRECTED per clinical review)

### 7. Diabetes - Level 2
- **Effect**: Elevated thresholds at BOTH low and high frequencies
  - 250 Hz: +8 dB
  - 500 Hz: +7 dB
  - 1000 Hz: +5 dB
  - 2000 Hz: +5 dB
  - 4000 Hz: +7 dB
  - 8000 Hz: +10 dB
- **Source**: NHANES (Bainbridge et al. 2008)
- **Status**: ✅ Implemented in simulation; placeholder in-situ

### 8. Cardiovascular Risk - Level 2 ⚠️ CORRECTED
- **Effect**: LOW frequency loss (apex of cochlea poorly perfused)
  - 250 Hz: +10 dB (most affected)
  - 500 Hz: +8 dB
  - 1000 Hz: +5 dB
  - 2000 Hz: +3 dB
  - 4000 Hz: +2 dB
  - 8000 Hz: +2 dB
- **Source**: Tan et al. 2023 + clinical expert (RK)
- **Status**: ✅ Implemented in simulation (CORRECTED); placeholder in-situ

### 9. Meniere's Disease - Level 2 ⚠️ CORRECTED
- **Effect**: LOW frequency loss + INCREASED UNCERTAINTY
  - 250 Hz: +30 dB (most affected)
  - 500 Hz: +25 dB
  - 1000 Hz: +15 dB
  - 2000 Hz: +10 dB
  - 4000 Hz: +5 dB
  - 8000 Hz: +5 dB
  - Variance factor: 2.0× (fluctuating thresholds)
- **Source**: ASHA JSLHR 42(4):829, PMC12250289 + clinical (RK)
- **Status**: ✅ Implemented in simulation; placeholder in-situ

### 10. Ototoxic Medication - Level 2
- **Effect**: HIGH frequency ski-slope loss
  - 250 Hz: 0 dB
  - 500 Hz: 0 dB
  - 1000 Hz: +2 dB
  - 2000 Hz: +5 dB
  - 4000 Hz: +15 dB (onset)
  - 6000 Hz: +25 dB (severe)
  - 8000 Hz: +35 dB (most severe)
- **Source**: Clinical (RK)
- **Status**: ✅ Implemented in simulation; placeholder in-situ

### 11. Vertigo/Balance ⚠️ CLARIFIED
- **If Meniere's etiology**: Use Meniere's conditioning (low-freq)
- **If vestibular schwannoma**: HIGH frequency ski-slope
  - 250 Hz: 0 dB
  - 500 Hz: +5 dB
  - 1000 Hz: +10 dB
  - 2000 Hz: +20 dB
  - 4000 Hz: +35 dB
  - 8000 Hz: +55 dB
- **Unknown etiology**: Increase uncertainty only (variance ×1.5)
- **Status**: ✅ Implemented in simulation; placeholder in-situ

---

## Psychometric Parameter Priors (Response Model)

These affect the psychometric function (likelihood model), not threshold priors:

### False Positive Rate (α)
- **Base**: Beta(1.5, 28.5), mean = 0.05
- **Tinnitus adjustment**: +0.05 within ±1 octave of tinnitus pitch
- **Source**: Schaette et al. 2011
- **Status**: ✅ Implemented in simulation

### False Negative Rate (β)
- **Base**: Beta(1, 49), mean = 0.02
- **Cognitive impairment**: +0.03
- **Source**: Kim et al. 2023
- **Status**: ✅ Implemented in simulation

### Psychometric Slope (σ)
- **Base**: LogNormal(log(8), 0.3), mean ≈ 8 dB
- **First audiogram**: +2 dB (increased uncertainty)
- **Cochlear hearing loss**: narrower (5-8 dB)
- **Status**: ✅ Implemented in simulation

### Response Window
- **Default**: 3000 ms
- **Cognitive concerns**: 5000 ms
- **Status**: ✅ Implemented in simulation

---

## Stimulus Parameters (Bayesian-controlled)

| Parameter | Bayesian PTA | Manual mHW |
|-----------|--------------|------------|
| Tone duration | 1.0-3.0s adaptive | Fixed 1-2s |
| Inter-stimulus interval | 1-3s variable + catch trials | Random 1-3s |
| Catch trials | For real-time α estimation | None |
| Temporal integration (τf) | Duration-adjusted thresholds | Not modeled |

**Status**: ✅ Specified in manuscript; simulation implementation pending

---

## File Locations

### In-Silico Simulation Codebase
- Prior construction: `audiometry_ai/priors/nhanes_priors.py`
- Conditioning functions: `audiometry_ai/priors/conditioning.py`
- NHANES data acquisition: `audiometry_ai/data/nhanes_downloader.py`
- Build script: `scripts/build_nhanes_priors.py`

### In-Situ Clinical System
- Prior conditioning service: `src/services/priorConditioningService.ts`

---

## Processing Order

1. NHANES-derived base prior (age-sex stratified) - Level 1
2. Risk factor adjustments - Level 2:
   a. Diabetes
   b. Cardiovascular risk (LOW freq)
   c. Noise exposure (4 kHz notch)
   d. Meniere's (LOW freq + uncertainty)
   e. Ototoxicity (HIGH freq ski-slope)
   f. Vertigo (depends on etiology)
3. Tympanometry conditioning - Level 3
4. Previous audiogram (if available) - strongest influence

---

---

## NHANES Empirical Validation (January 2025)

Empirical analysis of NHANES 2015-2018 data (n=7,713) with **age-adjusted regression**:

Model: `threshold ~ age + age² + male + diabetes + hypertension`

### Key Findings

| Parameter | Forecast vs Empirical | Significance | Action |
|-----------|----------------------|--------------|--------|
| **Presbycusis** | Forecast underestimates by ~2-3× | -- | Increase dB/year to ~1.2 |
| **Sex (male)** | Forecast underestimates at HF | p<0.001 at 6-8 kHz | Increase to +10-14 dB |
| **Diabetes** | Forecast over-estimates | None significant | Reduce effects, add uncertainty |
| **CV risk** | Pattern validated (low-freq) | +7.9 dB at 500 Hz | Keep low-freq emphasis |

### Age-Adjusted Results Summary

**Sex (Male disadvantage):**
- 3000 Hz: +10.3 dB (p=0.01)
- 4000 Hz: +9.6 dB (p=0.001)
- 6000 Hz: +13.8 dB (p<0.001)
- 8000 Hz: +12.3 dB (p<0.001)

**Diabetes:** No significant effects after age adjustment (p>0.5 all frequencies)

**Hypertension/CV:** Low-frequency emphasis supported
- 500 Hz: +7.9 dB (matches forecast of +8 dB)
- 8000 Hz: -9.2 dB (p=0.06, near-significant negative)

See `docs/nhanes_vs_forecast_comparison.md` for full analysis.

---

*Document updated: January 2025*
*Status: Post-clinical review (RK), NHANES validation complete*
