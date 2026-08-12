import math
import unittest
from io import BytesIO
from pathlib import Path

from docx import Document

from report_builder import build_practical_report

from thermal_lab_core import (
    analyse_conduction,
    analyse_radiation,
    assigned_online_conduction_data,
    assigned_online_radiation_data,
    blank_conduction_data,
    conduction_uncertainty_components,
    demonstration_conduction_data,
    demonstration_radiation_data,
    equilibrium_sensor_temperature_C,
    forced_convection_h,
    radiation_corrected_medium_temperature_C,
)


class ThermalLabCoreTests(unittest.TestCase):
    def test_streamlit_editor_dtypes_are_stable(self):
        blank = blank_conduction_data()
        demonstration = demonstration_conduction_data()
        self.assertEqual(str(blank["Trial"].dtype), "string")
        self.assertEqual(str(demonstration["Trial"].dtype), "string")
        self.assertTrue(all(str(blank[column].dtype) == "float64" for column in blank.columns[2:]))

    def test_online_assignments_are_stable_and_student_specific(self):
        first = assigned_online_conduction_data("12345678")
        repeat = assigned_online_conduction_data("12345678")
        second = assigned_online_conduction_data("87654321")
        self.assertTrue(first.equals(repeat))
        self.assertFalse(first.equals(second))
        self.assertEqual(len(analyse_conduction(first)), 6)

        radiation_first = assigned_online_radiation_data("12345678")
        radiation_repeat = assigned_online_radiation_data("12345678")
        self.assertTrue(radiation_first.equals(radiation_repeat))
        self.assertEqual(len(analyse_radiation(radiation_first)), 4)

    def test_controlled_conduction_data_has_expected_material_properties(self):
        data = demonstration_conduction_data()
        analysed = analyse_conduction(data, diameter_mm=25.0, heat_rate_fraction=1.0)
        brass = analysed[analysed["Material"] == "Brass"]
        aluminium = analysed[analysed["Material"] == "Aluminium"]
        self.assertTrue(brass["Thermal_conductivity_W_mK"].between(110.0, 128.0).all())
        self.assertTrue(aluminium["Thermal_conductivity_W_mK"].between(162.0, 198.0).all())
        self.assertTrue((analysed["Hot_contact_Rpp_m2K_W"] > 0).all())
        self.assertTrue((analysed["Cold_contact_Rpp_m2K_W"] > 0).all())
        self.assertTrue((analysed["Cold_contact_jump_K"] > 2.5 * analysed["Hot_contact_jump_K"]).all())
        self.assertTrue((analysed["Quality_flags"] == "No automatic flags").all())

    def test_all_online_conduction_variants_remain_in_reference_ranges(self):
        for student_id in ("00000000", "00000001", "00000002", "00000003", "12345678", "87654321"):
            analysed = analyse_conduction(assigned_online_conduction_data(student_id))
            brass = analysed[analysed["Material"] == "Brass"]
            aluminium = analysed[analysed["Material"] == "Aluminium"]
            self.assertTrue(brass["Thermal_conductivity_W_mK"].between(110.0, 128.0).all())
            self.assertTrue(aluminium["Thermal_conductivity_W_mK"].between(162.0, 198.0).all())

    def test_contact_temperatures_match_the_three_fitted_lines(self):
        analysed = analyse_conduction(demonstration_conduction_data())
        row = analysed.iloc[0]
        hot_interface = 0.0375
        cold_interface = 0.0675
        self.assertAlmostEqual(
            row["Hot_bar_face_C"],
            row["Hot_slope_K_m"] * hot_interface + row["Hot_intercept_C"],
            places=10,
        )
        self.assertAlmostEqual(
            row["Sample_hot_face_C"],
            row["Sample_slope_K_m"] * hot_interface + row["Sample_intercept_C"],
            places=10,
        )
        self.assertAlmostEqual(
            row["Sample_cold_face_C"],
            row["Sample_slope_K_m"] * cold_interface + row["Sample_intercept_C"],
            places=10,
        )
        self.assertAlmostEqual(
            row["Cold_bar_face_C"],
            row["Cold_slope_K_m"] * cold_interface + row["Cold_intercept_C"],
            places=10,
        )

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
        natural_shielded = analysed.iloc[1]
        forced_exposed = analysed.iloc[2]
        forced_shielded = analysed.iloc[3]
        self.assertGreater(natural_exposed["T8_error_K"], natural_exposed["T7_error_K"])
        self.assertGreater(natural_exposed["T9_error_K"], natural_exposed["T8_error_K"])
        self.assertTrue(
            (natural_shielded[["T7_error_K", "T8_error_K", "T9_error_K"]]
             < natural_exposed[["T7_error_K", "T8_error_K", "T9_error_K"]]).all()
        )
        self.assertTrue(
            (forced_exposed[["T7_error_K", "T8_error_K", "T9_error_K"]]
             < natural_exposed[["T7_error_K", "T8_error_K", "T9_error_K"]]).all()
        )
        self.assertTrue(
            (forced_shielded[["T7_error_K", "T8_error_K", "T9_error_K"]]
             < forced_exposed[["T7_error_K", "T8_error_K", "T9_error_K"]]).all()
        )
        self.assertTrue(math.isfinite(float(analysed["Maximum_abs_error_K"].max())))

    def test_all_online_radiation_variants_preserve_the_teaching_hierarchy(self):
        error_columns = ["T7_error_K", "T8_error_K", "T9_error_K"]
        for student_id in ("00000000", "00000001", "00000002", "00000003", "12345678", "87654321"):
            analysed = analyse_radiation(assigned_online_radiation_data(student_id))
            natural_exposed, natural_shielded, forced_exposed, forced_shielded = [analysed.iloc[index] for index in range(4)]
            self.assertGreater(natural_exposed["T9_error_K"], natural_exposed["T8_error_K"])
            self.assertGreater(natural_exposed["T8_error_K"], natural_exposed["T7_error_K"])
            self.assertTrue((natural_shielded[error_columns] < natural_exposed[error_columns]).all())
            self.assertTrue((forced_exposed[error_columns] < natural_exposed[error_columns]).all())
            self.assertTrue((forced_shielded[error_columns] < forced_exposed[error_columns]).all())

    def test_conduction_uncertainty_components_reproduce_rss_total(self):
        components = conduction_uncertainty_components(
            voltage=10.0,
            current=1.0,
            delta_temperature=2.5,
            diameter_mm=25.0,
            spacing_mm=15.0,
            voltage_uncertainty=0.01,
            current_uncertainty=0.01,
            temperature_uncertainty=0.10,
            diameter_uncertainty_mm=0.10,
            spacing_uncertainty_mm=0.10,
        )
        expected = math.sqrt(sum(value**2 for key, value in components.items() if key != "Combined"))
        self.assertAlmostEqual(components["Combined"], expected, places=12)
        self.assertGreater(components["Temperature difference"], components["Voltage"])

    def test_word_practical_report_is_generated_and_readable(self):
        raw = demonstration_conduction_data()
        analysed = analyse_conduction(raw)
        report = build_practical_report(
            practical_code="conduction",
            practical_title="Contact resistance in linear heat conduction",
            student_details={
                "name": "QA Student",
                "student_id": "12345678",
                "group": "QA",
                "lab_date": "2026-08-12",
                "pathway": "Online simulated practical",
            },
            raw_data=raw,
            analysed_data=analysed,
            aim="Verify the report generator.",
            equations=[("Q = VI", "Electrical heat input")],
            parameter_definitions=[("Q", "Heat rate; unit: W")],
            assumptions=[("Diameter", "25 mm")],
            sample_calculation=["Q = 7.0 x 0.70 = 4.90 W."],
            evidence=["The controlled teaching data are physically consistent."],
            discussion_notes=[("Temperature distribution", "Three fitted regions and two contact jumps are visible.")],
            logo_path=Path(__file__).resolve().parent / "assets" / "jcu_logo.jpg",
        )
        self.assertGreater(len(report), 20_000)
        document = Document(BytesIO(report))
        all_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn("Contact resistance in linear heat conduction", all_text)
        self.assertNotIn("THERMALLAB PRACTICAL REPORT", all_text)
        self.assertIn("Key graphs", all_text)
        self.assertIn("Sample calculation", all_text)


if __name__ == "__main__":
    unittest.main()
