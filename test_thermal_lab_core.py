import math
import unittest

from thermal_lab_core import (
    analyse_conduction,
    analyse_radiation,
    demonstration_conduction_data,
    demonstration_radiation_data,
    equilibrium_sensor_temperature_C,
    forced_convection_h,
    radiation_corrected_medium_temperature_C,
)


class ThermalLabCoreTests(unittest.TestCase):
    def test_supplied_conduction_example_recalculates_from_raw_values(self):
        data = demonstration_conduction_data()
        analysed = analyse_conduction(data, diameter_mm=25.0, heat_rate_fraction=1.0)
        brass_12_5 = analysed[(analysed["Material"] == "Brass") & (analysed["Voltage_V"] == 12.5)].iloc[0]
        self.assertAlmostEqual(brass_12_5["Hot_bar_face_C"], 86.9, delta=0.06)
        self.assertAlmostEqual(brass_12_5["Sample_hot_face_C"], 83.8, delta=0.06)
        self.assertAlmostEqual(brass_12_5["Sample_cold_face_C"], 79.8, delta=0.06)
        self.assertAlmostEqual(brass_12_5["Cold_bar_face_C"], 27.3, delta=0.08)
        self.assertAlmostEqual(brass_12_5["Hot_contact_Rpp_m2K_W"], 9.817e-5, delta=9.817e-5 * 0.02)
        self.assertAlmostEqual(brass_12_5["Cold_contact_Rpp_m2K_W"], 1.650e-3, delta=1.650e-3 * 0.02)

    def test_forced_convection_h_tracks_supplied_theoretical_scale(self):
        h_small, re_small, _ = forced_convection_h(4.0, 0.5)
        h_large, re_large, _ = forced_convection_h(4.0, 3.0)
        self.assertTrue(380 < h_small < 470)
        self.assertTrue(120 < h_large < 180)
        self.assertGreater(re_large, re_small)

    def test_sensor_forward_and_inverse_models_are_consistent(self):
        t_air = 25.0
        t_wall = 120.0
        h = 30.0
        emissivity = 0.90
        t_sensor = equilibrium_sensor_temperature_C(t_air, t_wall, h, emissivity)
        recovered_air = radiation_corrected_medium_temperature_C(t_sensor, t_wall, h, emissivity)
        self.assertTrue(t_air < t_sensor < t_wall)
        self.assertAlmostEqual(recovered_air, t_air, delta=1e-7)

    def test_supplied_radiation_example_has_expected_bias_pattern(self):
        analysed = analyse_radiation(demonstration_radiation_data())
        natural_exposed = analysed.iloc[0]
        forced_exposed = analysed.iloc[2]
        self.assertGreater(natural_exposed["T8_error_K"], natural_exposed["T7_error_K"])
        self.assertLess(forced_exposed["Maximum_abs_error_K"], natural_exposed["Maximum_abs_error_K"])
        self.assertTrue(math.isfinite(float(analysed["Maximum_abs_error_K"].max())))


if __name__ == "__main__":
    unittest.main()
