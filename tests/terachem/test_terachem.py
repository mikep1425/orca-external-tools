"""
Integration tests for the TeraChem wrapper.

The reference values below were produced with a development version of TeraChem (v1.9) of
the Bannwarth group. Other versions may not reproduce them to the tolerances used here;
please contact the author in that case.

Author: Mike Pauls
"""

import shutil
import unittest
from pathlib import Path

from oet.core.test_utilities import (
    OH,
    WATER,
    get_filenames,
    read_result_file,
    run_wrapper,
    write_input_file,
    write_xyz_file,
)

# Get the path to the script that should be tested
resolved_terachem_script = shutil.which("oet_terachem")
if resolved_terachem_script is None:
    raise RuntimeError(
        "The 'oet_terachem' script was not found in PATH. "
        "Run the tests with the project's virtual environment activated."
    )
terachem_script_path = Path(resolved_terachem_script)

# Leave terachem_executable_path empty, if terachem from system path should be called
terachem_executable_path = ""

# TeraChem template the reference values were produced with. The method is the only
# setting that differs between the test cases, since the open-shell case requires an
# unrestricted method. The keywords taken from the ORCA input ('coordinates', 'charge',
# 'spinmult' and 'run') are deliberately absent: the wrapper appends them.
TEMPLATE = """\
method                  {method}
basis                   def2-svp
max_l_in_basis          2
sphericalbasis          yes
guess                   sad
dftgrid                 5
purify                  no

threall                 1e-14
precision               double
"""

# TeraChem needs considerably longer than the default of the test utilities
TIMEOUT = 600.0


def write_template(filename: str | Path, method: str) -> None:
    with open(filename, "w") as f:
        f.write(TEMPLATE.format(method=method))


def run_terachem(inputfile: str, output_file: str, template: str | None = None) -> None:
    arguments = []
    if terachem_executable_path:
        arguments += ["--exe", terachem_executable_path]
    if template:
        arguments += ["--template", template]
    run_wrapper(
        inputfile=inputfile,
        script_path=terachem_script_path,
        outfile=output_file,
        args=arguments or None,
        timeout=TIMEOUT,
    )


class TerachemTests(unittest.TestCase):
    def test_H2O_engrad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O")
        write_xyz_file(xyz_file, WATER)
        write_template("tc.inp", "pbe")
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_terachem(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -76.2724575673
        expected_gradients = [
            0.0069215213,
            0.0051230225,
            -0.0036211094,
            -0.0030556832,
            -0.0039535030,
            0.0006154624,
            -0.0038658381,
            -0.0011695195,
            0.0030056470,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=6)

    def test_OH_anion_eng_grad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_anion")
        write_xyz_file(xyz_file, OH)
        # Also covers locating the template via the command line instead of the default name
        template = "OH_anion.template.inp"
        write_template(template, "pbe")
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=-1,
            multiplicity=1,
            ncores=2,
            do_gradient=1,
        )
        run_terachem(input_file, output_file, template=template)

        expected_num_atoms = 2
        expected_energy = -75.5746124647
        expected_gradients = [
            0.0066643030,
            0.0214476040,
            0.0061100791,
            -0.0066643030,
            -0.0214476040,
            -0.0061100791,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=6)

    def test_OH_rad_eng_grad(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("OH_rad")
        write_xyz_file(xyz_file, OH)
        write_template("tc.inp", "upbe")
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=2,
            ncores=2,
            do_gradient=1,
        )
        run_terachem(input_file, output_file)

        expected_num_atoms = 2
        expected_energy = -75.5811568880
        expected_gradients = [
            0.0055010174,
            0.0177040157,
            0.0050435144,
            -0.0055010174,
            -0.0177040157,
            -0.0050435144,
        ]

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        for g1, g2 in zip(gradients, expected_gradients):
            self.assertAlmostEqual(g1, g2, places=6)

    def test_H2O_energy_only(self):
        xyz_file, input_file, engrad_out, output_file = get_filenames("H2O_energy")
        write_xyz_file(xyz_file, WATER)
        write_template("tc.inp", "pbe")
        write_input_file(
            filename=input_file,
            xyz_filename=xyz_file,
            charge=0,
            multiplicity=1,
            ncores=2,
            do_gradient=0,
        )
        run_terachem(input_file, output_file)

        expected_num_atoms = 3
        expected_energy = -76.2724575673

        try:
            num_atoms, energy, gradients = read_result_file(engrad_out)
        except Exception as e:
            raise FileNotFoundError(
                f"Error wrapper outputfile not found. Check {output_file} for details"
            ) from e

        self.assertEqual(num_atoms, expected_num_atoms)
        self.assertAlmostEqual(energy, expected_energy, places=7)
        self.assertEqual(gradients, [])


if __name__ == "__main__":
    unittest.main()
