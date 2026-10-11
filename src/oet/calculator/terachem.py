#!/usr/bin/env python3
"""
This is a simple wrapper for the TeraChem binary (www.petachem.com), compatible with ORCA's
ExtTool interface.

TeraChem is configured via its own input file, so this wrapper does not generate the method
setup itself. Instead, it takes a user-provided template (by default 'tc.inp' next to the ORCA
input) and appends the keywords that follow from the ORCA input: 'coordinates', 'charge',
'spinmult' and 'run'. Any of these keywords present in the template is discarded, since ORCA
is authoritative for them. Note that the template is responsible for requesting an
unrestricted method for open-shell systems.

Parallelization
---------------
TeraChem runs on the GPU, and the devices it uses are set in the template ('gpus'), not by
this wrapper. The NCores given by ORCA is therefore only used to cap the CPU-side threading
via OMP_NUM_THREADS, so that the calculation stays within the allocation ORCA was given.

As a consequence, this wrapper cannot prevent device contention. Job types that run several
external calls at the same time (e.g. a NumFreq run with '%pal nprocs > 1') start one
TeraChem process per worker, and all of them read the same template and hence target the
same device. Their files do not collide, since every call runs in its own temporary
directory, but the calculations compete for the GPU. To serialize them, start the
calculation server with a core budget matching the ORCA input,

    oet_server terachem -n <nprocs>

and use 'oet_client' as ProgExt: every job requests NCores from that budget, so only one
runs at a time.

Provides
--------
class: TerachemCalc(BaseCalc)
    Class for performing a TeraChem calculation together with ORCA
main: function
    Main function
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import (
    check_file,
    check_path,
    nat_from_xyzfile,
    run_command,
)

# Default name of the TeraChem template input
TEMPLATE_NAME = "tc.inp"

# Keywords that are set from the ORCA input and are therefore dropped from the template
CONTROLLED_KEYWORDS = ("coordinates", "charge", "spinmult", "run")

# Header of the gradient table in the TeraChem output
GRADIENT_HEADER = "dE/dX"


class TerachemCalc(BaseCalc):
    @property
    def PROGRAM_NAMES(self) -> list[str]:
        """Program names to search for in PATH"""
        return ["terachem", "otool_terachem"]

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add TeraChem parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument("-e", "--exe", dest="prog", help="Path to the TeraChem executable")
        parser.add_argument(
            "-t",
            "--template",
            dest="template",
            help=f"Path to the TeraChem template input (default: '{TEMPLATE_NAME}' "
            "in the directory of the ORCA input)",
        )

    def find_template(self, calc_data: CalculationData, template: str | None) -> Path:
        """
        Locate the TeraChem template input. The path given on the command line takes
        precedence, otherwise the default name is looked up in the directory of the ORCA
        input. The calculation itself runs in a temporary directory, so the template cannot
        be picked up from the current working directory.

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        template: str | None
            Path given on the command line. None if no path was given

        Returns
        -------
        Path
            Path to the template input

        Raises
        ------
        FileNotFoundError
            If no template was found
        """
        if template:
            template_path = Path(template).expanduser()
            # The calculation runs in a temporary directory, so a relative path has to be
            # resolved against the directory of the ORCA input, just like the default name.
            if not template_path.is_absolute():
                template_path = calc_data.orca_input_dir / template_path
            template_path = template_path.resolve()
            if not check_file(template_path):
                raise FileNotFoundError(f"TeraChem template not found: {template_path}")
            return template_path
        template_path = (calc_data.orca_input_dir / TEMPLATE_NAME).resolve()
        if not check_file(template_path):
            raise FileNotFoundError(
                f"No TeraChem template '{TEMPLATE_NAME}' found in {calc_data.orca_input_dir}. "
                "Provide one there or pass its path with '--template'"
            )
        return template_path

    def write_terachem_input(
        self,
        calc_data: CalculationData,
        template_path: Path,
        terachem_inp: str | Path,
    ) -> None:
        """
        Write the TeraChem input from the template and the settings of the ORCA input.

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        template_path: Path
            Template input provided by the user
        terachem_inp: str | Path
            File to write the TeraChem input to
        """
        with template_path.open() as f:
            # Only the first token of a line is a keyword, so comments and values that happen
            # to contain one of the controlled keywords are kept.
            template_lines = [
                line
                for line in f
                if not line.split() or line.split()[0].lower() not in CONTROLLED_KEYWORDS
            ]

        run_type = "gradient" if calc_data.dograd else "energy"
        with Path(terachem_inp).open("w") as f:
            f.writelines(template_lines)
            f.write(f"\ncoordinates    {calc_data.xyzfile.name}\n")
            f.write(f"charge         {calc_data.charge}\n")
            f.write(f"spinmult       {calc_data.mult}\n")
            f.write(f"run            {run_type}\n")

    @staticmethod
    def first_float(line: str) -> float | None:
        """
        Return the first field of a line that can be read as a number, None if there is none.

        Parameters
        ----------
        line: str
            Line to read from

        Returns
        -------
        float | None
            The first numerical field of the line
        """
        for field in line.split():
            try:
                return float(field)
            except ValueError:
                continue
        return None

    def read_terachem_out(
        self, calc_data: CalculationData, natoms: int
    ) -> tuple[float, list[float]]:
        """
        Read energy and gradient from the TeraChem output. The energy is taken from the
        'FINAL ENERGY' line, the gradient from the table that follows the 'dE/dX' header and
        holds one line of three components (Eh/Bohr) per atom:

            Gradient units are Hartree/Bohr
            ---------------------------------------------------
                    dE/dX            dE/dY            dE/dZ
                0.0069215213     0.0051230225    -0.0036211094
                ...
            ---------------------------------------------------

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        natoms: int
            Number of atoms in the system

        Returns
        -------
        float
            The computed energy
        list[float]
            The gradient (X,Y,Z) for each atom

        Raises
        ------
        ValueError
            If the energy is missing or the gradient is incomplete
        """
        energy = None
        gradient: list[float] = []
        reading_gradient = False
        output_path = check_path(calc_data.output_file)
        with output_path.open() as f:
            for line in f:
                if energy is None and "FINAL" in line and "ENERGY" in line.upper():
                    energy = self.first_float(line)
                elif GRADIENT_HEADER in line:
                    reading_gradient = True
                elif reading_gradient:
                    fields = line.split()
                    if len(fields) == 3:
                        gradient += [float(i) for i in fields]
                    elif gradient:
                        # The table is closed by a dashed line
                        reading_gradient = False

        if energy is None:
            raise ValueError(f"Total energy not found in file {calc_data.output_file}")
        if calc_data.dograd and len(gradient) != 3 * natoms:
            raise ValueError(
                f"Number of gradient entries: {len(gradient)} does not match 3x number of atoms: {natoms}"
            )
        return energy, gradient

    def run_terachem(
        self,
        calc_data: CalculationData,
        terachem_inp: str | Path,
        args: list[str],
    ) -> None:
        """
        Run the TeraChem program with the generated input file and redirect its STDOUT and
        STDERR to a logfile.

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        terachem_inp: str | Path
            The generated TeraChem input file
        args: list[str]
            Arguments not parsed so far
        """
        # This does not parallelize the calculation, which runs on the GPU set in the
        # template, but caps the CPU-side threading to the allocation given by ORCA.
        # MKL falls back to OMP_NUM_THREADS, so it needs no separate variable.
        os.environ["OMP_NUM_THREADS"] = str(calc_data.ncores)

        if not calc_data.prog_path:
            raise RuntimeError("Path to program is None.")
        run_command(calc_data.prog_path, calc_data.output_file, args + [str(terachem_inp)])

    def calc(
        self, calc_data: CalculationData, args_parsed: Namespace, args_not_parsed: list[str]
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        calc_data: CalculationData
            Object with calculation data for the run
        args_parsed: Namespace
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parsed so far

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty
        """
        # Get parsed options
        prog = args_parsed.prog
        # Set and check the program path if its executable
        calc_data.set_program_path(prog)
        if calc_data.prog_path:
            print(f"Using executable {calc_data.prog_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid executable from standard program names: {self.PROGRAM_NAMES}"
            )

        # Point charges would have to be passed to TeraChem explicitly, so fail loudly
        # instead of silently computing the system without them.
        if calc_data.pointcharges:
            raise ValueError("Point charges are not supported by this wrapper")

        # Template input and name for the generated input
        template_path = self.find_template(calc_data=calc_data, template=args_parsed.template)
        print(f"Using TeraChem template {template_path}")
        terachem_inp = f"{calc_data.basename}.tc.inp"

        # Write the TeraChem input
        self.write_terachem_input(
            calc_data=calc_data,
            template_path=template_path,
            terachem_inp=terachem_inp,
        )

        # run TeraChem
        self.run_terachem(
            calc_data=calc_data,
            terachem_inp=terachem_inp,
            args=args_not_parsed,
        )

        # get the number of atoms from the xyz file
        natoms = nat_from_xyzfile(xyz_file=calc_data.xyzfile)

        # parse the TeraChem output
        energy, gradient = self.read_terachem_out(
            calc_data=calc_data,
            natoms=natoms,
        )

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = TerachemCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()
