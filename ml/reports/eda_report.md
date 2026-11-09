# Exploratory Data Analysis Report
## Solar Power Generation and Performance Deviation Analysis-II

### 1. Dataset Overview
The dataset contains operational telemetry and meteorological data collected from two solar power plants over a 34-day span from **May 15, 2020** to **June 17, 2020** at 15-minute sampling intervals.

| Dataset File | Records | Columns | Unique Inverters / Sensors | Unique Timestamps | Date Range |
|---|---|---|---|---|---|
| Plant_1_Generation_Data.csv | 68,778 | 7 | 22 inverters | 3,158 | 2020-05-15 00:00:00 to 2020-06-17 23:45:00 |
| Plant_1_Weather_Sensor_Data.csv | 3,182 | 6 | 1 weather sensor | 3,182 | 2020-05-15 00:00:00 to 2020-06-17 23:45:00 |
| Plant_2_Generation_Data.csv | 67,698 | 7 | 22 inverters | 3,259 | 2020-05-15 00:00:00 to 2020-06-17 23:45:00 |
| Plant_2_Weather_Sensor_Data.csv | 3,259 | 6 | 1 weather sensor | 3,259 | 2020-05-15 00:00:00 to 2020-06-17 23:45:00 |

### 2. Data Quality & Integrity Audit
- **Missing Values**: Zero null or missing cells across all 4 raw CSV files.
- **Duplicate Records**: Zero duplicate rows found.
- **Timestamp Formats**:
  - `Plant_1_Generation_Data.csv` format: `%d-%m-%Y %H:%M` (day first).
  - `Plant_1_Weather_Sensor_Data.csv` format: `%Y-%m-%d %H:%M:%S`.
  - `Plant_2_Generation_Data.csv` format: `%Y-%m-%d %H:%M:%S`.
  - `Plant_2_Weather_Sensor_Data.csv` format: `%Y-%m-%d %H:%M:%S`.
  - *Engineering Note*: Using unconstrained mixed date parsing causes Plant 1 dates (`01-06-2020` to `12-06-2020`) to be parsed as January-December instead of June 1 to June 12. Format-aware parsing ensures 100% temporal alignment.
- **Sampling Continuity**:
  - Expected interval: 15 minutes.
  - Plant 1 Weather continuity: 99.7% regular. 8 gaps (max gap: 0 days 07:15:00).
  - Plant 2 Weather continuity: 99.8% regular. 5 gaps (max gap: 0 days 00:30:00).
- **Physical Range Validation**:
  - Irradiation: [0.000, 1.222] kW/m² in Plant 1; [0.000, 1.099] kW/m² in Plant 2. Physically realistic.
  - Ambient Temperature: [20.4°C, 35.3°C] in Plant 1; [20.9°C, 39.2°C] in Plant 2.
  - Module Temperature: [18.1°C, 65.5°C] in Plant 1; [20.3°C, 66.6°C] in Plant 2.

### 3. Statistical Summaries

#### Plant 1 Aggregated (Plant Level):
- **Mean AC Power**: 6704.83 kW (Std: 8604.22 kW, Max: 29150.21 kW)
- **Daytime Mean AC Power (Irradiation > 0)**: 12067.93 kW
- **Mean Module Temp**: 31.18°C (Max: 65.55°C)
- **Mean Ambient Temp**: 25.56°C (Max: 35.25°C)

#### Plant 2 Aggregated (Plant Level):
- **Mean AC Power**: 5011.97 kW (Std: 6317.87 kW, Max: 25979.76 kW)
- **Daytime Mean AC Power (Irradiation > 0)**: 8772.29 kW
- **Mean Module Temp**: 32.77°C (Max: 66.64°C)
- **Mean Ambient Temp**: 28.07°C (Max: 39.18°C)

### 4. Correlation Analysis
Correlation with target `AC_POWER`:
- **IRRADIATION**: r = 0.996 (Plant 1), r = 0.911 (Plant 2). Dominant linear driver.
- **MODULE_TEMPERATURE**: r = 0.961 (Plant 1), r = 0.872 (Plant 2).
- **AMBIENT_TEMPERATURE**: r = 0.726 (Plant 1), r = 0.648 (Plant 2).

### 5. Nighttime & Zero-Generation Analysis
- Between 18:30 and 05:45, solar irradiation drops to 0.0 kW/m² and AC generation drops to exactly 0 kW across all inverters.
- Nighttime records account for ~53.8% of all 15-minute observations.
- *Treatment Decision*: A model that predicts small non-zero values (e.g. 10 W or -5 W) at 2:00 AM violates basic solar physics. We incorporate solar threshold gating where if $Irradiation = 0$ and $Hour \in [19, 5]$, predicted power is deterministically 0 kW. This avoids distortion of daytime regression loss while maintaining zero nighttime error.

### 6. Plant-Wise Operational Differences
1. **DC-to-AC Scaling Ratio**:
   - In Plant 1, DC power is ~10x AC power (e.g. DC ~6,000 to AC ~600).
   - In Plant 2, DC power and AC power are on similar numeric scales (e.g. DC ~400 to AC ~390).
   - This indicates different inverter conversion telemetry definitions between the two plants.
2. **Thermal Response**:
   - Plant 2 experiences higher ambient temperatures (up to 39.2°C vs 35.3°C in Plant 1).
   - Module temperature reaches up to 66.6°C, causing noticeable thermal efficiency derating at peak midday solar hours.
3. **Inverter Variance**:
   - In Plant 1, inverter generation is remarkably uniform across all 22 units (~6.8M to ~7.8M total yield).
   - In Plant 2, several inverters show significantly lower cumulative generation (e.g. `Quc1TzRxBtWmjKH` and `EtZseqShD0Zmfok`), indicating localized outages or degradation.

### 7. Important Observations and Limitations
1. **Short Observation Window**: The dataset spans only 34 days (late spring / early summer). It does not contain autumn or winter seasonality, solar equinox variations, or annual degradation. Models trained on this dataset must clearly document that they do not guarantee year-round generalizability.
2. **Target Granularity**: While inverter-level prediction allows tracking individual inverter failures, weather sensors are recorded at the plant level (one central station per plant). Aggregated plant-level modeling creates a direct physical mapping from central weather parameters to plant generation. We provide unified modeling supporting both plant-level and inverter-level capabilities.
