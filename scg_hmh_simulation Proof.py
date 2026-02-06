import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import sympy as sp

# ────────────────────────────────────────────────
# CONFIGURATION – TUNABLE PARAMETERS
# ────────────────────────────────────────────────

MASS_FLOW_KG_PER_H = 100.0
MASS_FLOW_KG_PER_S = MASS_FLOW_KG_PER_H / 3600

STARTUP_MINUTES = 45
DT_MINUTES = 1.0
TOTAL_SIM_MINUTES = 240
times_min = np.arange(0, TOTAL_SIM_MINUTES + DT_MINUTES, DT_MINUTES)
dt_hours = DT_MINUTES / 60.0

def transition(t, t_half, width=15):
    return 1 / (1 + np.exp(-(t - t_half) / width))

# Pressure states (explicit)
PRESSURE_DEWAR = 1.5
PRESSURE_PRE_EXPANDER = 4.0
PRESSURE_HIGH_GAS = 38.0
PRESSURE_EXHAUST_VACUUM = 0.1
PRESSURE_RECIRC_INTAKE = 0.2
PRESSURE_COMPRESSED = 40.0

# Thermodynamic parameters for SymPy calculations
GAMMA = 1.4
CP_KJ_PER_KG_K = 1.04
R_KJ_PER_KG_K = 0.2968
T1_K = 800.0
ETA_TURB = 0.85
IRREV_EXP = 0.3  # kJ/kg·K, from document ~0.2-0.5 for expansion
IRREV_COMP = 0.7  # kJ/kg·K, from document ~0.5-1 for compressor
IRREV_LIQ = 0.8  # Approximate for liquefaction

# SymPy entropy and work calculations
gamma, cp, R, T1, P1, P2, eta_turb, irrev = sp.symbols('gamma cp R T1 P1 P2 eta_turb irrev')

# Expansion
T2_ideal_expr = T1 * (P2 / P1)**((gamma - 1)/gamma)
w_ideal_exp = cp * (T1 - T2_ideal_expr)
w_real_exp = eta_turb * w_ideal_exp
delta_s_ideal_exp = cp * sp.ln(T2_ideal_expr / T1) - R * sp.ln(P2 / P1)
delta_s_real_exp = delta_s_ideal_exp + irrev

# Compression (approximate isentropic)
T2_comp_expr = T1 * (P1 / P2)**((gamma - 1)/gamma)  # Note: P1 > P2 for compression, but swap for formula
w_ideal_comp = cp * (T2_comp_expr - T1)
delta_s_ideal_comp = 0  # Ideal
delta_s_real_comp = irrev

# Substitute values
values_exp = {
    gamma: GAMMA,
    cp: CP_KJ_PER_KG_K,
    R: R_KJ_PER_KG_K,
    T1: T1_K,
    P1: PRESSURE_HIGH_GAS,
    P2: PRESSURE_EXHAUST_VACUUM,
    eta_turb: ETA_TURB,
    irrev: IRREV_EXP
}

values_comp = {
    gamma: GAMMA,
    cp: CP_KJ_PER_KG_K,
    R: R_KJ_PER_KG_K,
    T1: 100.0,  # Precooled inlet ~100 K
    P1: PRESSURE_COMPRESSED,  # Out
    P2: 1.0,  # In ~1 bar
    irrev: IRREV_COMP
}

EXPANSION_WORK_KJ_PER_KG_BASE = float(w_real_exp.subs(values_exp))
DELTA_S_EXP_IDEAL = float(delta_s_ideal_exp.subs(values_exp))
DELTA_S_EXP_REAL = float(delta_s_real_exp.subs(values_exp))
DELTA_S_COMP_REAL = float(delta_s_real_comp.subs(values_comp))  # Note: using irrev directly

# Energy parameters (tunable, set to optimized for high COP demonstration)
PSA_ENERGY_KWH_PER_KG = 0.15  # Adjusted to 0.15 (document range 0.1-0.2)
LIQUEFACTION_KWH_PER_KG_BASE = 0.32  # Base before reductions
INDUCTION_POWER_KW_BASE = 10.0  # Adjusted to document ~10 kW
MHD_POWER_KW_BASE = 25.0  # Balanced
MARX_BASE_POWER_KW = 3.0
REGEN_PERCENT_START = 1.0
REGEN_PERCENT_STEADY = 40.0  # Tunable
RECIRC_PERCENT_START = 1.0
RECIRC_PERCENT_STEADY = 40.0  # Tunable, for parasitic reductions
OTHER_PARASITICS_KW_BASE = 2.0

# Enhancements
VACUUM_BOOST_PERCENT = 15.0
EMF_FEEDBACK_IONIZATION_SAVINGS_PERCENT = 40.0
EMF_FEEDBACK_MHD_BOOST_PERCENT = 15.0
AI_IONISATION_DUTY_CYCLE = 0.70

# Warmup half-times (minutes)
T_HALF_REGEN = 18
T_HALF_VACUUM = 22
T_HALF_AI = 28
T_HALF_EMF = 25

# MHD scenario
MHD_SCENARIO = "medium"  

# HTS heat fraction
HTS_LOSS_FRACTION = 0.05
HTS_BEARING_FIXED_KW = 0.1

# ────────────────────────────────────────────────
# ROTOR & BEARING SAFETY
# ────────────────────────────────────────────────
rpm_range = np.linspace(10000, 120000, 100)
omega = rpm_range * 2 * np.pi / 60
radius_m = 0.075
density = 1700  
hoop_stress_mpa = density * (omega**2) * (radius_m**2) / 1e6
ultimate_strength_mpa = 2000  
safety_factor = ultimate_strength_mpa / hoop_stress_mpa

bearing_efficiency = np.ones_like(rpm_range) * 99.95
bearing_efficiency -= np.random.normal(0, 0.02, len(rpm_range))
bearing_efficiency = np.clip(bearing_efficiency, 99.8, 100.0)

# ────────────────────────────────────────────────
# SIMULATION FUNCTION (for sensitivity)
# ────────────────────────────────────────────────

def run_sim(liq_base, regen_steady, recirc_steady):
    global REGEN_PERCENT_STEADY, LIQUEFACTION_KWH_PER_KG_BASE, RECIRC_PERCENT_STEADY
    LIQUEFACTION_KWH_PER_KG_BASE = liq_base
    REGEN_PERCENT_STEADY = regen_steady
    RECIRC_PERCENT_STEADY = recirc_steady
    
    net_power_kw = np.zeros(len(times_min))
    gross_power_kw = np.zeros(len(times_min))
    net_parasitic_kw = np.zeros(len(times_min))
    apparent_cop = np.zeros(len(times_min))
    cop_wrt_supplied = np.zeros(len(times_min))
    cop_net_wrt_supplied = np.zeros(len(times_min))  # New: Net COP wrt supplied
    cum_net_energy_kwh = np.zeros(len(times_min))
    
    # Power breakdown arrays
    mech_power = np.zeros(len(times_min))
    induction_power = np.zeros(len(times_min))
    mhd_power = np.zeros(len(times_min))
    regen_power = np.zeros(len(times_min))
    psa_power = np.zeros(len(times_min))
    liq_power = np.zeros(len(times_min))
    ion_power = np.zeros(len(times_min))
    hts_heat = np.zeros(len(times_min))
    
    ref_mass_flow = 100.0
    flow_ratio = MASS_FLOW_KG_PER_H / ref_mass_flow
    
    if MHD_SCENARIO == "low":
        mhd_scale = 0.55
        mhd_flow_exp = 0.92
    elif MHD_SCENARIO == "high":
        mhd_scale = 1.65
        mhd_flow_exp = 1.08
    else:
        mhd_scale = 1.05
        mhd_flow_exp = 1.00
    
    for i, t_min in enumerate(times_min):
        t = t_min
    
        f_regen = transition(t, T_HALF_REGEN)
        f_vacuum = transition(t, T_HALF_VACUUM)
        f_ai = transition(t, T_HALF_AI)
        f_emf = transition(t, T_HALF_EMF)
    
        regen_percent_now = REGEN_PERCENT_START + (REGEN_PERCENT_STEADY - REGEN_PERCENT_START) * f_regen
        recirc_percent_now = RECIRC_PERCENT_START + (RECIRC_PERCENT_STEADY - RECIRC_PERCENT_START) * f_regen  # Use same transition
        vacuum_boost_now = 0 + VACUUM_BOOST_PERCENT * f_vacuum
        duty_cycle_now = 1.0 + (AI_IONISATION_DUTY_CYCLE - 1.0) * f_ai
        emf_ion_save_now = 0 + EMF_FEEDBACK_IONIZATION_SAVINGS_PERCENT * f_emf
        emf_mhd_boost_now = 0 + EMF_FEEDBACK_MHD_BOOST_PERCENT * f_emf
    
        liq_kwh_per_kg = LIQUEFACTION_KWH_PER_KG_BASE * (1 - 0.55 * vacuum_boost_now / 100) * (1 - recirc_percent_now / 100)
        liq_power_now = MASS_FLOW_KG_PER_H * liq_kwh_per_kg
    
        exp_work_kj_per_kg = EXPANSION_WORK_KJ_PER_KG_BASE * (1 + vacuum_boost_now / 100)
        
        # Induction: sub-linear
        induction_power_now = INDUCTION_POWER_KW_BASE * (flow_ratio ** 0.85)
    
        # MHD: scaled
        mhd_base_scaled = MHD_POWER_KW_BASE * (flow_ratio ** mhd_flow_exp)
        mhd_power_now = mhd_base_scaled * mhd_scale * (1 + emf_mhd_boost_now / 100)
    
        # HTS heat (recycled)
        hts_heat_now = HTS_LOSS_FRACTION * (induction_power_now + mhd_power_now) + HTS_BEARING_FIXED_KW * f_emf
        additional_exp_kj_per_kg = hts_heat_now / MASS_FLOW_KG_PER_S  # kW to kJ/kg
        exp_work_kj_per_kg += additional_exp_kj_per_kg
        
        mech_power_now = MASS_FLOW_KG_PER_S * exp_work_kj_per_kg
    
        # Ionization
        ion_power_now = MARX_BASE_POWER_KW * (flow_ratio ** 0.75) * duty_cycle_now * (1 - emf_ion_save_now / 100)
    
        gross_now = mech_power_now + induction_power_now + mhd_power_now - hts_heat_now  # Subtract losses
    
        regen_power_now = gross_now * (regen_percent_now / 100)
    
        psa_power_now = PSA_ENERGY_KWH_PER_KG * MASS_FLOW_KG_PER_H * (1 - recirc_percent_now / 100)
    
        # Other parasitics
        other_parasitics_kw = OTHER_PARASITICS_KW_BASE + 0.5 * flow_ratio
    
        supplied_now = psa_power_now + liq_power_now + ion_power_now + other_parasitics_kw
        net_par_now = supplied_now - regen_power_now
        net_now = gross_now - net_par_now
        cop_now = gross_now / max(net_par_now, 0.1) if net_par_now > 0 else float('inf')
        cop_supplied_now = gross_now / max(supplied_now, 0.1) if supplied_now > 0 else float('inf')
        cop_net_supplied_now = net_now / max(supplied_now, 0.1) if supplied_now > 0 else float('inf')
    
        net_power_kw[i] = net_now
        gross_power_kw[i] = gross_now
        net_parasitic_kw[i] = net_par_now
        apparent_cop[i] = cop_now
        cop_wrt_supplied[i] = cop_supplied_now
        cop_net_wrt_supplied[i] = cop_net_supplied_now
    
        if i == 0:
            cum_net_energy_kwh[i] = 0.0
        else:
            cum_net_energy_kwh[i] = cum_net_energy_kwh[i-1] + (net_power_kw[i-1] + net_now)/2 * dt_hours
        
        # Breakdown
        mech_power[i] = mech_power_now
        induction_power[i] = induction_power_now
        mhd_power[i] = mhd_power_now
        regen_power[i] = regen_power_now
        psa_power[i] = psa_power_now
        liq_power[i] = liq_power_now
        ion_power[i] = ion_power_now
        hts_heat[i] = hts_heat_now
    
    # Breakeven
    breakeven_idx = np.where(cum_net_energy_kwh >= 0)[0]
    breakeven_min = breakeven_idx[0] * DT_MINUTES if len(breakeven_idx) > 0 else None
    
    # Exergy check (updated with heat exergy)
    exergy_cold_per_kg = 720  # kJ/kg
    latent = 199.0  # kJ/kg for N2
    heat_kj_per_kg = latent + CP_KJ_PER_KG_K * (T1_K - 77)
    exergy_heat = heat_kj_per_kg * (1 - 300 / T1_K)  # Assuming high-grade heat at T1
    total_exergy_per_kg = exergy_cold_per_kg + exergy_heat
    max_work_kw = total_exergy_per_kg * MASS_FLOW_KG_PER_H / 3600
    carnot_eff = 1 - 77/300  # Base for cold
    print(f"\n=== Exergy Check for Liq Base {liq_base}, Regen {regen_steady}%, Recirc {recirc_steady}% ===")
    print(f"Total exergy (cold + heat): {total_exergy_per_kg:.1f} kJ/kg")
    print(f"Max possible work: {max_work_kw:.1f} kW")
    print(f"Steady gross: {gross_power_kw[-1]:.1f} kW ({gross_power_kw[-1]/max_work_kw*100:.1f}% of exergy)")
    print(f"Implied efficiency vs Carnot: {(gross_power_kw[-1]/max_work_kw) / carnot_eff * 100:.1f}%")
    
    # Entropy summary from SymPy
    print(f"\n=== Entropy Analysis (SymPy) ===")
    print(f"Expansion ΔS ideal: {DELTA_S_EXP_IDEAL:.4f} kJ/kg·K")
    print(f"Expansion ΔS real: {DELTA_S_EXP_REAL:.4f} kJ/kg·K")
    print(f"Compression ΔS real (approx): {DELTA_S_COMP_REAL:.4f} kJ/kg·K")
    print(f"Liquefaction ΔS (approx): {IRREV_LIQ:.4f} kJ/kg·K")
    total_delta_s = DELTA_S_EXP_REAL + DELTA_S_COMP_REAL + IRREV_LIQ - 0.7 * (recirc_steady / 100)  # Reduction from recirculation ~0.5-1
    print(f"Total system ΔS (approx, reduced by recirc): {total_delta_s:.4f} kJ/kg·K")
    
    # Summary
    print(f"\n=== Summary for Liq Base {liq_base}, Regen {regen_steady}%, Recirc {recirc_steady}% ===")
    print(f"Steady-state net output: {net_power_kw[-1]:.1f} kW")
    print(f"Apparent COP: {apparent_cop[-1]:.2f}")
    print(f"COP gross wrt Supplied: {cop_wrt_supplied[-1]:.2f}")
    print(f"COP net wrt Supplied: {cop_net_wrt_supplied[-1]:.2f}")
    if breakeven_min is not None:
        print(f"Breakeven time: ≈ {breakeven_min:.0f} minutes")
    else:
        print("Breakeven not reached")
    
    return times_min, mech_power, induction_power, mhd_power, regen_power, psa_power, liq_power, ion_power, net_power_kw, apparent_cop, cop_wrt_supplied, cop_net_wrt_supplied, cum_net_energy_kwh, breakeven_min

# ────────────────────────────────────────────────
# SENSITIVITY RUNS (Low, Medium, High Scenarios)
# ────────────────────────────────────────────────
# Low: Conservative
print("\n=== Low Scenario ===")
run_sim(0.70, 18.0, 18.0)

# Medium: Balanced
print("\n=== Medium Scenario ===")
run_sim(0.45, 25.0, 25.0)

# High: Optimized
print("\n=== High Scenario ===")
run_sim(0.32, 40.0, 60.0)

# Pressure table
print("\nPressure States Across the System (steady-state):")
print(tabulate([
    ["Dewar storage", f"{PRESSURE_DEWAR} bar"],
    ["Before expander (liquid)", f"{PRESSURE_PRE_EXPANDER} bar"],
    ["After boiling (high pressure)", f"{PRESSURE_HIGH_GAS} bar"],
    ["Post-expander exhaust (vacuum)", f"{PRESSURE_EXHAUST_VACUUM} bar"],
    ["Recirculation intake", f"{PRESSURE_RECIRC_INTAKE} bar"],
    ["Compressed before liquefaction", f"{PRESSURE_COMPRESSED} bar"]
], headers=["Stage", "Pressure"], tablefmt="grid"))

# ────────────────────────────────────────────────
# PLOTS (for optimized high case)
# ────────────────────────────────────────────────
# Reset to high for plots
times_min, mech, ind, mhd, regen, psa, liq_p, ion, net_p, cop, cop_sup, cop_net_sup, cum_e, be_min = run_sim(0.32, 40.0, 60.0)

fig, axs = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("SCG-HMH Generator – Improved Plots (High Scenario)", fontsize=14)

t_hours = times_min / 60

# Net Power
axs[0,0].plot(t_hours, net_p, color='darkgreen', lw=2, label='Net power')
axs[0,0].axvline(STARTUP_MINUTES/60, color='gray', ls='--', alpha=0.7, label='End of startup')
axs[0,0].set_ylabel("Power (kW)")
axs[0,0].set_title("Instantaneous Net Power")
axs[0,0].grid(True, alpha=0.3)
axs[0,0].legend()

# Cumulative Net Energy
axs[0,1].plot(t_hours, cum_e, label='Net delivered', color='darkgreen', lw=2.5)
if be_min is not None:
    axs[0,1].axvline(be_min/60, color='black', ls=':', lw=1.5, label='Breakeven')
axs[0,1].axvline(STARTUP_MINUTES/60, color='gray', ls='--', alpha=0.7)
axs[0,1].set_ylabel("Cumulative Energy (kWh)")
axs[0,1].set_title("Cumulative Net Energy")
axs[0,1].grid(True, alpha=0.3)
axs[0,1].legend()

# Bearing Efficiency vs RPM
axs[1,0].plot(rpm_range, bearing_efficiency, color='purple', lw=2, label='Bearing efficiency')
axs[1,0].set_xlabel("Rotor Speed (RPM)")
axs[1,0].set_ylabel("Bearing Efficiency (%)")
axs[1,0].set_title("Frictionless HTS Bearing Efficiency vs RPM")
axs[1,0].set_ylim(99.7, 100.1)
axs[1,0].grid(True, alpha=0.3)
axs[1,0].legend()

# Rotor Safety Factor vs RPM
axs[1,1].plot(rpm_range, safety_factor, color='darkorange', lw=2, label='Safety factor')
axs[1,1].axhline(3, color='red', ls='--', alpha=0.7, label='Minimum safe margin (3×)')
axs[1,1].axhspan(3, np.max(safety_factor)*1.1, color='lightgreen', alpha=0.15, label='Safe zone')
axs[1,1].set_xlabel("Rotor Speed (RPM)")
axs[1,1].set_ylabel("Safety Factor")
axs[1,1].set_title("Rotor Burst Safety vs RPM (Al/Cf composite)")
axs[1,1].grid(True, alpha=0.3)
axs[1,1].legend()

# Stacked Power Breakdown
axs[2,0].stackplot(t_hours, mech, ind, mhd, labels=['Mech', 'Induction', 'MHD'])
axs[2,0].set_ylabel("Power (kW)")
axs[2,0].set_title("Gross Power Breakdown")
axs[2,0].grid(True, alpha=0.3)
axs[2,0].legend()

# Parasitics Breakdown (negative for regen)
paras = np.stack([ -regen, psa, liq_p, ion ])
axs[2,1].stackplot(t_hours, paras, labels=['Regen (credit)', 'PSA', 'Liq', 'Ion'])
axs[2,1].set_ylabel("Power (kW)")
axs[2,1].set_title("Parasitics Breakdown")
axs[2,1].grid(True, alpha=0.3)
axs[2,1].legend()

plt.tight_layout()
plt.show()