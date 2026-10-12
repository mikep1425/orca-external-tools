#!/usr/bin/env python3
"""
In-process tblite wrapper built on the tblite Python API, compatible with the
ExtTool interface.

Where TbliteCalc spawns the tblite executable once per gradient, this calculator
calls libtblite in-process. That matters in server mode (``oet_server tblite_api``),
where the worker process keeps the calculator instance alive, so that

* the GFN parametrisation is built once instead of per call, and
* no tblite process is started per call.

Reusing the previous call's converged density as the SCF guess is available via
--restart but OFF by default: how much it saves depends on how far the geometry
moved between calls and so on the system and the search, and at larger
displacements it saves nothing at all.

Standalone use (``oet_tblite_api``) pays the full tblite import on every call --
about 205 ms, of which ~160 ms is numpy, which the tblite bindings require and
import themselves -- and is therefore much SLOWER than TbliteCalc. It exists for
testing and for checking results against the subprocess path, not for production
runs. A server worker pays that import once at start-up instead.

Provides
--------
class: TbliteApiCalc(BaseCalc)
    Calculator that evaluates energy and gradient through the tblite Python API
main: function
    Main function
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import numpy as np
from tblite.interface import Calculator

from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import (
    LENGTH_CONVERSION,
    mult_to_nue,
    symbols_to_numbers,
    xyzfile_to_at_coord,
)

# ORCA writes coordinates in Angstrom, tblite expects Bohr
ANG_TO_BOHR = 1.0 / LENGTH_CONVERSION["Ang"]

# Command-line method names mapped to tblite parametrisation names
METHODS: dict[str, str] = {
    "gfn1": "GFN1-xTB",
    "gfn2": "GFN2-xTB",
    "ipea1": "IPEA1-xTB",
}

# Maximum number of (Calculator, Result) pairs held per worker process.
# Each concurrent GOAT worker runs in its own directory and so gets its own entry;
# the bound keeps a long run from growing the cache without limit.
MAX_CACHED_CALCULATORS = 64


class TbliteApiCalc(BaseCalc):
    """Energy and gradient from tblite, evaluated in-process."""

    # The structure is read straight into memory, so no scratch directory is needed
    needs_tmpdir = False

    def __init__(self) -> None:
        super().__init__()
        # Cache of calculators and their last result, keyed so that concurrent GOAT
        # workers never inherit each other's density guess (see _cache_key).
        self._cache: OrderedDict[tuple[Any, ...], tuple[Calculator, Any]] = OrderedDict()

    @property
    def PROGRAM_NAMES(self) -> list[str]:
        """No external executable is called, so nothing is searched for in PATH."""
        return []

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """
        Add tblite API options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument(
            "--method",
            dest="method",
            default="gfn2",
            choices=sorted(METHODS),
            help="xTB parametrisation to use. Default: gfn2",
        )
        parser.add_argument(
            "--accuracy",
            dest="accuracy",
            type=float,
            default=None,
            help="Numerical SCC threshold. Default: tblite's own (1.0)",
        )
        parser.add_argument(
            "--max-iter",
            dest="max_iter",
            type=int,
            default=None,
            help="Maximum number of SCC iterations. Default: tblite's own (250)",
        )
        parser.add_argument(
            "--restart",
            dest="restart",
            action="store_true",
            help="Reuse the converged density of the previous call in the same "
            "directory as the SCF guess. Only has an effect in server mode, where a "
            "previous result exists. Off by default: the saving depends on how far "
            "the geometry moved and is nil for large displacements.",
        )

    @staticmethod
    def _settings(args_parsed: Namespace) -> tuple[tuple[str, Any], ...]:
        """
        Collect the tblite ``set`` options that were requested.

        Returned as a sorted tuple so that it can be part of the cache key: a
        calculator built with different settings must not be reused.

        Parameters
        ----------
        args_parsed: Namespace
            Arguments parsed as defined in extend_parser

        Returns
        -------
        tuple[tuple[str, Any], ...]
            (attribute, value) pairs to pass to Calculator.set
        """
        settings: dict[str, Any] = {}
        if args_parsed.accuracy is not None:
            settings["accuracy"] = args_parsed.accuracy
        if args_parsed.max_iter is not None:
            settings["max-iter"] = args_parsed.max_iter
        return tuple(sorted(settings.items()))

    def _cache_key(
        self,
        calc_data: CalculationData,
        numbers: np.ndarray,
        uhf: int,
        args_parsed: Namespace,
    ) -> tuple[Any, ...]:
        """
        Build the cache key for a calculator and its stored density.

        The ORCA input directory is part of the key on purpose. GOAT runs several
        workers concurrently, each in its own directory and each following its own
        trajectory, and the server hands a request to whichever worker process is
        free. Keying on the directory keeps every trajectory's density separate;
        without it a worker would be handed a guess from an unrelated geometry,
        which costs more SCC iterations than starting cold.

        Parameters
        ----------
        calc_data: CalculationData
            Calculation data
        numbers: np.ndarray
            Atomic numbers
        uhf: int
            Number of unpaired electrons
        args_parsed: Namespace
            Arguments parsed as defined in extend_parser

        Returns
        -------
        tuple[Any, ...]: The cache key
        """
        return (
            str(calc_data.orca_input_dir),
            numbers.tobytes(),
            calc_data.charge,
            uhf,
            args_parsed.method,
            self._settings(args_parsed),
        )

    def calc(
        self, calc_data: CalculationData, args_parsed: Namespace, args_not_parsed: list[str]
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient through the tblite API.

        Parameters
        ----------
        calc_data: CalculationData
            Calculation data
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

        Raises
        ------
        RuntimeError
            If unknown arguments were passed
        """
        if args_not_parsed:
            raise RuntimeError(
                f"Unknown arguments for the tblite API calculator: {' '.join(args_not_parsed)}"
            )

        # tblite is parallelised with OpenMP and has no API option for the thread
        # count. MKL_NUM_THREADS is set alongside OMP_NUM_THREADS because libtblite
        # may be linked against a threaded MKL (libmkl_intel_thread), which takes
        # MKL_NUM_THREADS in preference to OMP_NUM_THREADS and would otherwise
        # thread independently of what ORCA asked for.
        #
        # NOTE: this is authoritative only for standalone use. Both runtimes sample
        # these variables when they initialise, so in server mode -- where the
        # library is already loaded -- the pinning must be in the environment
        # BEFORE oet_server starts:
        #     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 oet_server tblite_api ...
        os.environ["OMP_NUM_THREADS"] = str(calc_data.ncores)
        os.environ["MKL_NUM_THREADS"] = str(calc_data.ncores)

        symbols, coords_ang = xyzfile_to_at_coord(calc_data.xyzfile)
        numbers = np.asarray(symbols_to_numbers(symbols))
        positions = np.asarray(coords_ang, dtype=float) * ANG_TO_BOHR
        uhf = mult_to_nue(calc_data.mult)

        key = self._cache_key(calc_data, numbers, uhf, args_parsed)
        entry = self._cache.get(key)
        if entry is None:
            calculator = Calculator(
                METHODS[args_parsed.method],
                numbers,
                positions,
                charge=float(calc_data.charge),
                uhf=uhf,
            )
            for attribute, value in self._settings(args_parsed):
                calculator.set(attribute, value)
            # Results are returned as arrays, so tblite's own printout is noise
            calculator.set("verbosity", 0)
            previous = None
        else:
            calculator, previous = entry
            calculator.update(positions=positions)

        # Passing the previous result reuses its density as the SCF guess.
        # It is updated in place and returned, so the cache holds the same object.
        # Without --restart the calculator is still reused (the parametrisation is
        # the expensive part); only the density guess is discarded.
        result = calculator.singlepoint(previous if args_parsed.restart else None)

        self._cache[key] = (calculator, result)
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CACHED_CALCULATORS:
            self._cache.popitem(last=False)

        energy = float(result.get("energy"))
        gradient: list[float] = []
        if calc_data.dograd:
            # (natoms, 3) in Eh/Bohr, flattened to A1X, A1Y, A1Z, A2X, ...
            grad = np.asarray(result.get("gradient"), dtype=float)
            if grad.shape != (calc_data.natoms, 3):
                raise RuntimeError(
                    f"Gradient shape {grad.shape} does not match "
                    f"({calc_data.natoms}, 3) expected from the structure"
                )
            gradient = grad.ravel().tolist()

        reused = "reused density" if entry is not None and args_parsed.restart else "fresh density"
        print(
            f"tblite {METHODS[args_parsed.method]} in-process ({reused}): "
            f"total energy {energy:.10f} Eh"
        )
        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = TbliteApiCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()
