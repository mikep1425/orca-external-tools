# tblite (in-process API)

`oet_tblite_api` evaluates GFN-xTB energies and gradients through the
[tblite Python API](https://tblite.readthedocs.io/en/latest/api/python.html)
instead of spawning the `tblite` executable per call, which is what
`oet_tblite` does.

This only pays off behind `oet_server`. Standalone, `oet_tblite_api` imports the
tblite bindings on every invocation -- ~205 ms, of which ~160 ms is numpy, a hard
requirement of the bindings (they reject plain lists) rather than an oet choice --
and is therefore much **slower** than `oet_tblite`. It exists for testing and for
cross-checking results against the subprocess path. A server worker pays that
import once at start-up.

Install with the optional dependency group:

```bash
pip install -e '.[tblite]'
```

> **If you built tblite yourself** (e.g. with meson against MKL), do **not** use
> this extra, and do not pass `--extra tblite` to `install.py`: it installs the
> `tblite` wheel from PyPI, which ships its own bundled library and will
> overwrite your bindings. Install oet without the extra and add `numpy`
> separately. The PyPI wheel is fine if you have no local build and do not care
> which BLAS it uses.

## Why it is faster in server mode

ORCA starts an external tool once per gradient, so a GOAT run pays every
per-call cost tens of thousands of times. With the calculator instance held
resident by a server worker:

* the GFN parametrisation is built once instead of per call,
* no `tblite` process is started per call, and
* no scratch directory is created, copied into and removed per call
  (`needs_tmpdir = False`).

## Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--method {gfn1,gfn2,ipea1}` | `gfn2` | xTB parametrisation. |
| `--accuracy X` | tblite's own (1.0) | Numerical SCC threshold. |
| `--max-iter N` | tblite's own (250) | Maximum SCC iterations. |
| `--restart` | off | Reuse the previous call's converged density in the same directory as the SCF guess. |

`--restart` is off by default on purpose. How much it saves depends on how far
the geometry moved since the previous call in that directory, so it is a
property of the system and of the search rather than a safe default: for small
displacements it cuts SCC iterations noticeably, and for large ones it saves
nothing. Measure it for your case before switching it on.

## Threads

tblite is parallelised with OpenMP, and `libtblite` may be linked against a
threaded MKL (`libmkl_intel_thread`), which takes `MKL_NUM_THREADS` in
preference to `OMP_NUM_THREADS`. Check which BLAS you have:

```bash
python -c "import tblite, pathlib; print(pathlib.Path(tblite.__file__).parent)"
ldd <that dir>/_libtblite*.so | grep -i -e mkl -e blas
```

Both runtimes sample their thread variables when they initialise, so in server
mode the pinning must be in the environment **before** the server starts --
setting it per request cannot re-thread an already-loaded library. For small
systems threading is actively counterproductive (at ~40 atoms a multi-threaded
single point can be an order of magnitude slower in wall time than a
single-threaded one), and GOAT gives each of its workers one core in any case.

## Server deployment (Slurm)

```bash
# Generate unique port for the job to avoid clashes between different servers
PORT=$((10000 + (SLURM_JOB_ID + ${SLURM_ARRAY_TASK_ID:-0}) % 50000))

# start local server where a calculator with tblite is available.
# The env prefix pins the calculator without touching ORCA's own OMP settings.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    oet_server tblite_api -n ${SLURM_NTASKS} -b localhost:${PORT} &

# Store the servers process ID to shut it down afterwards
SERVER_PID=$!
echo "Started Server on PORT ${PORT} with PID ${SERVER_PID} and $SLURM_NTASKS proc"

server_cleanup() {
    echo "Stopping server (PID ${SERVER_PID})"
    kill ${SERVER_PID} 2>/dev/null
    wait ${SERVER_PID} 2>/dev/null
}
trap 'server_cleanup' USR1 SIGTERM SIGKILL

# Substitute 'PORT' and 'TASKS' in the template
sed "s/PORT/${PORT}/g" orca.template > orca.inp
sed -i "s/TASKS/${SLURM_NTASKS}/g" orca.inp

# Wait for the server to get up
echo "Waiting for port ${PORT}..."
TIMEOUT=60
for i in $(seq 1 $TIMEOUT); do
    if nc -z localhost ${PORT}; then
        echo "Server is ready"
        break
    fi
    if [ "$i" -eq "$TIMEOUT" ]; then
        echo "ERROR: server did not come up within ${TIMEOUT}s" >&2
        server_cleanup
        exit 1
    fi
    sleep 1
done

# Run the orca calc. as usual
$ORCA orca.inp > orca.out

# IMPORTANT: Kill the server instance afterwards
server_cleanup
```

`orca.template`:

```
! Extopt goat
%method
 progext "/path/to/oet_client_c"
 ext_params "-b localhost:PORT --method gfn2"
end
%pal nprocs TASKS end
%maxcore 400
* xyzfile 0 1 coord.xyz
```

### Binding and node layout

`-b localhost:PORT` binds the loopback interface of the node the server runs on,
which is what you want while ORCA and the external tool run on that same node
(`#SBATCH --nodes=1`). `127.0.0.1` is equivalent and avoids an IPv6-first
resolution attempt, but either works.

If ORCA ever runs the external tool on a different node from the server, both
forms break -- the client would be addressing its own loopback. That needs
`-b 0.0.0.0:PORT` on the server and the server node's hostname (e.g.
`$SLURMD_NODENAME` captured before launch) in `ext_params`.

Note `make static` resolves numeric addresses only; with a hostname such as
`localhost`, use the default dynamic build. The difference is ~0.14 ms per call.

### Core budget

`-n ${SLURM_NTASKS}` is the server's total core budget, enforced by
`CoreLimiter` against the `NCores` each request asks for. GOAT splits its base
workers and may ask for more concurrent gradients than that budget allows; the
surplus queues rather than oversubscribing the node. Set it to the core count
you actually want busy.

## Choosing a client

`progext` can point at any of three clients. All three talk to the same
`/calculate` endpoint, so the server does not care which is used:

| Client | Per-call cost | Notes |
|--------|---------------|-------|
| `oet_client_c` (compiled, `src/oet/server_client/c_client`) | ~2 ms | Build with `make`; no dependencies beyond libc. |
| `oet_client_fast` | ~30 ms | Stdlib sockets only. Mostly interpreter start-up. |
| `oet_client` (Python, `client:client`) | ~90 ms | Imports `requests`; fine for slow calculators, not for xTB. |

For an xTB-class method the calculation itself is tens of milliseconds, so the
client's start-up is a first-order cost and the compiled client is the one to
use. For an MLIP, where a call costs far more, any of them will do.

```bash
make -C src/oet/server_client/c_client
make -C src/oet/server_client/c_client install PREFIX=$HOME/.local
```

## Checking results against the subprocess path

The API path and the executable path should agree to numerical noise. Worth
confirming on a real input before trusting any timings:

```bash
oet_tblite     orca_EXT.extinp.tmp --method gfn2 --exe $(which tblite)
mv orca_EXT.engrad ref.engrad
oet_tblite_api orca_EXT.extinp.tmp --method gfn2
diff ref.engrad orca_EXT.engrad
```
