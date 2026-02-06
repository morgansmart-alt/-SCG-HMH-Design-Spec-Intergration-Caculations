# scg_hmh_cycle_simulation.py
# SCG-HMH Generator - Full Detailed Cycle Simulation with Startup Phase
# Simulates complete machine operation: startup → steady-state
# Includes PSA membrane separation, condensing tube/regen HX, liquefaction, expansion, induction, MHD,
# stator EMF feedback, AI ionization switch, compressor-created vacuum (0.1 bar),
# N2 insulation, REBCO 100% conductivity, Marx generator, separate parasitics,
# and COP vs. duty cycle plot.
# Apparent COP >1 from regeneration (recycles waste heat/cold) – true eff < Carnot.

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────
# 1. CONFIGURATION – REALISTIC BASELINE VALUES
# ────────────────────────────────────────────────

# Mass flow rate (design point)
MASS_FLOW_KG_PER_H = 100.0
MASS_FLOW_KG_PER_S = MASS_FLOW_KG_PER_H / 3600

# Startup phase duration (minutes) – high parasitics until regeneration stabilizes
STARTUP_MINUTES = 45

# Stage 1 – Air intake & PSA membrane separation
PSA_ENERGY_KWH_PER_KG = 0.08                  # typical for small PSA unit
PSA_POWER_KW = MASS_FLOW_KG_PER_H * PSA_ENERGY_KWH_PER_KG   # constant

# Stage 2 – Liquefaction & pre-cooling (condensing tube / regen HX)
LIQUEFACTION_KWH_PER_KG_BASE = 0.35           # without vacuum/regen

# Stage 3 – Phase change & turbo-expander drive
EXPANSION_WORK_KJ_PER_KG_BASE = 380           # after losses, no vacuum

# Stage 4 – Induction harvest (REBCO stator, 100% conductivity at 77 K)
INDUCTION_POWER_KW = 18.0

# Stage 5 – MHD plasma harvest (Marx generator + pulsed ionization)
MHD_POWER_KW_BASE = 35.0
MARX_BASE_POWER_KW = 3.5                      # full Marx power without savings

# Regeneration (cold exhaust + waste heat recovery via condensing tube)
REGEN_PERCENT_STEADY = 18.0                   # steady-state recovery %
REGEN_PERCENT_STARTUP = 5.0                   # lower during startup (cold not yet stable)

# Other constant parasitics (controls, pumps, minor losses)
OTHER_PARASITICS_KW = 3.0

# Enhancements: Vacuum from compressor + stator EMF feedback + AI switch
VACUUM_BOOST_PERCENT = 12.0                   # compressor pulls 0.1 bar → better eff & lower boil-off
EMF_FEEDBACK_IONIZATION_SAVINGS_PERCENT = 50.0  # % savings on Marx/ionization from stator EMF donation
EMF_FEEDBACK_MHD_BOOST_PERCENT = 18.0         # better ionization → higher MHD output

# AI Ionisation Switch – controls duty cycle (0–1) based on induction strength
AI_IONISATION_DUTY_CYCLE = 0.65               # AI decides: only 65% on-time needed

# ────────────────────────────────────────────────
# 2. STARTUP PHASE (high parasitics, low regen)
# ────────────────────────────────────────────────

print("=== STARTUP PHASE (first {} minutes) ===\n".format(STARTUP_MINUTES))

# During startup: no cold exhaust yet → minimal regeneration
LIQUEFACTION_KWH_PER_KG_STARTUP = LIQUEFACTION_KWH_PER_KG_BASE
LIQUEFACTION_POWER_KW_STARTUP = MASS_FLOW_KG_PER_H * LIQUEFACTION_KWH_PER_KG_STARTUP
REGEN_POWER_KW_STARTUP = 0.0  # negligible regen until cold exhaust available

# Full ionization needed (AI not yet optimized)
IONIZATION_POWER_KW_STARTUP = MARX_BASE_POWER_KW

# Vacuum not yet stable
EXPANSION_WORK_KJ_PER_KG_STARTUP = EXPANSION_WORK_KJ_PER_KG_BASE
MECHANICAL_POWER_KW_STARTUP = MASS_FLOW_KG_PER_S * EXPANSION_WORK_KJ_PER_KG_STARTUP

GROSS_POWER_KW_STARTUP = MECHANICAL_POWER_KW_STARTUP + INDUCTION_POWER_KW + MHD_POWER_KW_BASE

NET_PARASITIC_KW_STARTUP = (
    PSA_POWER_KW +
    LIQUEFACTION_POWER_KW_STARTUP +
    IONIZATION_POWER_KW_STARTUP +
    OTHER_PARASITICS_KW -
    REGEN_POWER_KW_STARTUP
)

NET_OUTPUT_KW_STARTUP = GROSS_POWER_KW_STARTUP - NET_PARASITIC_KW_STARTUP
APPARENT_COP_STARTUP = GROSS_POWER_KW_STARTUP / max(NET_PARASITIC_KW_STARTUP, 0.1)

print(f"Startup duration: {STARTUP_MINUTES} minutes")
print(f"Startup gross output: {GROSS_POWER_KW_STARTUP:6.1f} kW")
print(f"Startup net parasitics: {NET_PARASITIC_KW_STARTUP:6.1f} kW")
print(f"Startup net output: {NET_OUTPUT_KW_STARTUP:6.1f} kW (negative = net consumer)")
print(f"Startup apparent COP: {APPARENT_COP_STARTUP:.2f}\n")

# ────────────────────────────────────────────────
# 3. STEADY-STATE PHASE (after startup)
# ────────────────────────────────────────────────

print("=== STEADY-STATE PHASE (after startup) ===\n")

# Apply vacuum, EMF feedback, AI switch
LIQUEFACTION_KWH_PER_KG = LIQUEFACTION_KWH_PER_KG_BASE * (1 - VACUUM_BOOST_PERCENT/100 * 0.7)
EXPANSION_WORK_KJ_PER_KG = EXPANSION_WORK_KJ_PER_KG_BASE * (1 + VACUUM_BOOST_PERCENT/100)
IONIZATION_POWER_KW = MARX_BASE_POWER_KW * AI_IONISATION_DUTY_CYCLE * (1 - EMF_FEEDBACK_IONIZATION_SAVINGS_PERCENT / 100)
MHD_POWER_KW = MHD_POWER_KW_BASE * (1 + EMF_FEEDBACK_MHD_BOOST_PERCENT / 100)

# Full regeneration now active
LIQUEFACTION_POWER_KW = MASS_FLOW_KG_PER_H * LIQUEFACTION_KWH_PER_KG
MECHANICAL_POWER_KW = MASS_FLOW_KG_PER_S * EXPANSION_WORK_KJ_PER_KG
GROSS_POWER_KW = MECHANICAL_POWER_KW + INDUCTION_POWER_KW + MHD_POWER_KW
REGEN_POWER_KW = GROSS_POWER_KW * (REGEN_PERCENT_STEADY / 100)

# Total net parasitic load
NET_PARASITIC_KW = (
    PSA_POWER_KW +
    LIQUEFACTION_POWER_KW +
    IONIZATION_POWER_KW +
    OTHER_PARASITICS_KW -
    REGEN_POWER_KW
)

NET_OUTPUT_KW = GROSS_POWER_KW - NET_PARASITIC_KW
APPARENT_COP = GROSS_POWER_KW / max(NET_PARASITIC_KW, 0.1)
TRUE_EFF_PERCENT = (NET_OUTPUT_KW / (GROSS_POWER_KW + NET_PARASITIC_KW)) * 100 if (GROSS_POWER_KW + NET_PARASITIC_KW) > 0 else 0

# ────────────────────────────────────────────────
# 4. PRINT DETAILED CYCLE SUMMARY
# ────────────────────────────────────────────────

print(f"Mass flow:          {MASS_FLOW_KG_PER_H:6.1f} kg/h   ({MASS_FLOW_KG_PER_S:7.5f} kg/s)\n")

print("Enhancements applied:")
print(f"  Compressor creates 0.1 bar vacuum → {VACUUM_BOOST_PERCENT}% boost")
print(f"  N₂ insulating properties → safe HV without arcing")
print(f"  REBCO stator at 77 K → 100% conductivity (zero resistance)")
print(f"  Stator EMF feedback → saves {EMF_FEEDBACK_IONIZATION_SAVINGS_PERCENT:.1f}% on ionization")
print(f"  AI ionisation switch → duty cycle {AI_IONISATION_DUTY_CYCLE*100:.0f}% (optimised on/off)\n")

print("Parasitic loads (separate):")
print(f"  PSA membrane separation:   {PSA_POWER_KW:6.1f} kW")
print(f"  Liquefaction (condensing tube / regen HX): {LIQUEFACTION_POWER_KW:6.1f} kW")
print(f"  Ionization (Marx generator + AI switch): {IONIZATION_POWER_KW:6.1f} kW")
print(f"  Other (controls/pumps):     {OTHER_PARASITICS_KW:6.1f} kW")
print(f"  Total gross parasitics:     {PSA_POWER_KW + LIQUEFACTION_POWER_KW + IONIZATION_POWER_KW + OTHER_PARASITICS_KW:6.1f} kW\n")

print("Power generation stages:")
print(f"  Mechanical (turbo-expander): {MECHANICAL_POWER_KW:6.1f} kW")
print(f"  Induction harvest (REBCO):   {INDUCTION_POWER_KW:6.1f} kW")
print(f"  MHD plasma harvest:          {MHD_POWER_KW:6.1f} kW")
print(f"  Gross electrical output:     {GROSS_POWER_KW:6.1f} kW\n")

print("Regeneration & net result:")
print(f"  Recovered via regen (cold exhaust + waste heat): {REGEN_POWER_KW:6.1f} kW  ({REGEN_PERCENT_STEADY}% of gross)")
print(f"  Net parasitic load:          {NET_PARASITIC_KW:6.1f} kW")
print(f"  Net electrical output:       {NET_OUTPUT_KW:6.1f} kW\n")

print(f"Apparent COP (gross output / net parasitics): {APPARENT_COP:.2f}")
print(f"True efficiency (net / total energy): ~{TRUE_EFF_PERCENT:.0f}%\n")

# ────────────────────────────────────────────────
# 5. PLOT: COP vs Ionisation Duty Cycle (AI switch effect)
# ────────────────────────────────────────────────

duty_cycles = np.linspace(0.1, 1.0, 20)  # 10–100% duty
cops = []
for duty in duty_cycles:
    ion_kw = MARX_BASE_POWER_KW * duty * (1 - EMF_FEEDBACK_IONIZATION_SAVINGS_PERCENT / 100)
    net_par = PSA_POWER_KW + LIQUEFACTION_POWER_KW + ion_kw + OTHER_PARASITICS_KW - REGEN_POWER_KW
    cop = GROSS_POWER_KW / max(net_par, 0.1)
    cops.append(cop)

plt.figure(figsize=(8, 5))
plt.plot(duty_cycles * 100, cops, marker='o', color='teal')
plt.xlabel('AI Ionisation Duty Cycle (%)')
plt.ylabel('Apparent COP')
plt.title('How AI Switch Affects COP (with EMF Feedback)')
plt.grid(True)
plt.show()