# Formula Student Vehicle Dynamics Simulator (V1.0)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/pygame-2.6%2B-green.svg)](https://www.pygame.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0%2B-orange.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)]()

A high-fidelity, interactive **Formula Student / FSAE Vehicle Dynamics Simulator** built from the ground up in Python. Designed for motorsport engineering students, simulation engineers, and racing enthusiasts, this platform simulates multi-body vehicle dynamics, tire mechanics, suspension kinematics, aerodynamic forces, and includes an intelligent AI Race Engineer and Neural Lap Predictor.

---

## Highlights & Key Features

### 🏎️ Vehicle Dynamics & Physics Engine
- **Non-Linear Tire Model**: Magic Formula / Pacejka tire modeling incorporating peak slip angle, cornering stiffness, slip ratio, and normal load sensitivity.
- **Dynamic Load Transfer**: Full 4-wheel independent normal load calculation under combined longitudinal braking/acceleration and lateral cornering forces.
- **Body Roll & Pitch Dynamics**: Roll center migration, front/rear roll stiffness distribution, pitch stiffness, and anti-roll bar (ARB) resistance.
- **Suspension Kinematics**: Quarter-car suspension model tracking spring rates, damping coefficients, motion ratios, bump travel, and wheel rates.

### 🏁 Track Simulator & Multi-Camera Rendering
- **Multiple Camera Views**:
  - **3D Chase Cam**: Dynamic trailing third-person view with pitch and yaw tracking.
  - **3D Cockpit Cam (FPV)**: Immersive first-person driver perspective with steering wheel and cockpit instrumentation.
  - **2D Top-Down / Tactical Cam**: Engineering bird's-eye view for racing line inspection and track mapping.
- **Ghost Car Replay**: Compare current laps against personal best reference ghosts in real time.
- **Speed Heatmap Overlay**: Dynamic track color-coding indicating high-speed straights vs corner apex speeds.
- **Split Telemetry & Wide Mode**: Seamlessly switch between dual-column engineering telemetry panels and full-screen racing visuals.

### 🧠 Intelligent Race Engineering Overlays (Nuclear HUD)
- **AI Race Engineer (`E`)**:
  - Live understeer/oversteer gradient analysis.
  - Aerodynamic balance and wake interaction alerts.
  - Dynamic damper and anti-roll bar tuning recommendations based on corner phase.
- **Neural Lap Predictor (`N`)**:
  - Frame-by-frame sector timing evaluation.
  - Machine-learning predicted lap times and theoretical optimal lap calculation.
  - Apex speed deltas and sector splits.
- **Aerodynamics Suite (`V`)**:
  - Front wing, rear wing, and ground-effect underfloor downforce computation.
  - Dynamic aerodynamic drag polar and DRS wake detection.

### 🔬 Engineering Labs & Diagnostic Modes
- **Design Lab (`L`)**: Real-time adjustment of vehicle geometry (wheelbase, track, mass distribution, CG height, tire friction coefficients).
- **Suspension Lab (`J`)**: Tune spring rates, damping rates, ARB stiffness, and bump compliance.
- **Handling Test (`H`)**: Automated steady-state cornering and skidpad testing for balance diagnosis.
- **Sensitivity Sweeps (`SPACE` in Design Lab)**: Parametric sweeps across vehicle configurations to plot handling characteristics.
- **Full Telemetry Studio (`T`)**: Live graphs for speed, G-G diagram, lateral/longitudinal acceleration, yaw rate, wheel loads, and slip angles.
- **Telemetry Data Export (`F`)**: Export high-frequency CSV logs suitable for external analysis (MoTeC, MATLAB, Excel).

---

## Keyboard Controls & Keybindings

### Global Navigation & Modes
| Key | Mode / Action | Description |
|---|---|---|
| `W` / `A` / `S` / `D` | Vehicle Controls | Throttle, Steering (Left/Right), and Brakes |
| `K` | **Track Simulator** | Launch the full racing circuit simulator |
| `L` | **Design Lab** | Enter vehicle parameters & chassis geometry editor |
| `J` | **Suspension Lab** | Enter suspension springs, dampers & ARB editor |
| `H` | **Handling Test** | Run automated steady-state skidpad handling analysis |
| `T` | **Telemetry Studio** | Open real-time telemetry graphs and G-G diagram |
| `R` | Reset | Reset vehicle state, lap counter, or tuning parameters |
| `ESC` / `ENTER` | Back / Drive Mode | Return to primary open testing drive ground |

### Track Simulator Controls (Inside Track Mode `K`)
| Key | Action | Description |
|---|---|---|
| `C` | Cycle Camera View | Toggle between **3D Chase**, **3D Cockpit FPV**, and **2D Top-Down** |
| `TAB` | Toggle Layout | Toggle between **Split Telemetry Panels** and **Wide-Screen View** |
| `G` | Toggle Ghost Car | Enable or disable personal-best ghost vehicle |
| `P` | Speed Heatmap | Render real-time color-gradient speed heatmap onto track |
| `F` | Export Telemetry | Export lap telemetry to timestamped CSV file (`lap_telemetry_*.csv`) |
| `V` | Aero HUD | Toggle Aerodynamics Simulation panel overlay |
| `E` | AI Engineer HUD | Toggle real-time AI Race Engineer diagnostics |
| `N` | Neural Predictor | Toggle Neural Lap Time Predictor & sector delta panel |
| `R` | Reset Lap | Reset current lap time and reposition car at start line |

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vimalkanth411/FORMULA_SIM.git
   cd FORMULA_SIM
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the simulator:**
   ```bash
   python main.py
   ```

---

## Technical Specifications

| Parameter | Default Value | Unit | Description |
|---|---|---|---|
| **Mass** | 250.0 | kg | Total vehicle mass including driver |
| **Wheelbase** | 1.55 | m | Distance between front and rear axle centers |
| **Track Width** | 1.20 | m | Distance between left and right wheel centerlines |
| **Weight Distribution** | 50 / 50 | % | Front-to-rear static weight distribution |
| **CG Height** | 0.30 | m | Center of gravity height above ground |
| **Front Spring Rate** | 30,000 | N/m | Front suspension spring rate |
| **Rear Spring Rate** | 32,000 | N/m | Rear suspension spring rate |
| **Front Damper** | 2,200 | N·s/m | Front suspension damping coefficient |
| **Rear Damper** | 2,400 | N·s/m | Rear suspension damping coefficient |
| **Front ARB** | 900.0 | N·m/rad | Front anti-roll bar torsional stiffness |
| **Rear ARB** | 750.0 | N·m/rad | Rear anti-roll bar torsional stiffness |
| **Tire Friction ($\mu$)** | 1.55 (F) / 1.50 (R) | - | Peak friction coefficient |
| **Peak Slip Angle** | 6.0 | deg | Slip angle at maximum lateral grip |

---

## Exported Telemetry Data Schema

When pressing `F` in Track Mode, high-frequency telemetry data is saved into a CSV file with the following channels:

| Column | Unit | Description |
|---|---|---|
| `time_s` | s | Elapsed lap time |
| `lap` | - | Lap counter |
| `progress` | 0.0 - 1.0 | Lap completion progress |
| `speed_kmh` | km/h | Vehicle velocity |
| `throttle` | 0.0 - 1.0 | Throttle pedal position |
| `brake` | 0.0 - 1.0 | Brake pedal position |
| `steering_deg` | deg | Front steering wheel angle |
| `yaw_rate_dps` | deg/s | Yaw angular velocity |
| `lat_g` | G | Lateral acceleration |
| `long_g` | G | Longitudinal acceleration |
| `lateral_error_m`| m | Distance deviation from optimal racing line |
| `sector` | 1 - 3 | Current track sector split |

---

## Project Structure

```
FORMULA_SIM/
├── main.py              # Core application: Physics engine, render loop, UI & HUD
├── requirements.txt     # Python package requirements (pygame, numpy)
├── .gitignore           # Git ignore rules for venv, cache, and telemetry logs
└── README.md            # Comprehensive project documentation
```

---

## Contributing

Contributions are welcome! If you would like to contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/CoolFeature`)
3. Commit your Changes (`git commit -m 'Add CoolFeature'`)
4. Push to the Branch (`git push origin feature/CoolFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
