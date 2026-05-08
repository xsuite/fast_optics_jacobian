from pathlib import Path

import numpy as np
import xtrack as xt
from prettytable import PrettyTable
import matplotlib.pyplot as plt


LATTICE_PATH = Path(__file__).parent / "lattice_data" / "fcc_lattice_corrected.json"

# Full-ring MAD-NG twiss is unstable for this FCC lattice in this setup.
# Use a stable open range with explicit initial conditions from the Xsuite seed.
START = "frf.1"
END = "qrdr6.1"
X_OFF = 0.5e-3


def _max_abs_diff(arr_a, arr_b):
    arr_a = np.asarray(arr_a)
    arr_b = np.asarray(arr_b)
    return float(np.max(np.abs(arr_a - arr_b)))


def _report_field(name, xs_data, ng_data, abs_tol):
    max_abs = _max_abs_diff(xs_data, ng_data)
    status = "OK" if max_abs <= abs_tol else "WARN"
    print(f"{name:12s} max|diff|={max_abs:.6e}   tol={abs_tol:.6e}   {status}")
    return status


def _compute_stats(xs_arr, ng_arr):
    xs = np.asarray(xs_arr, dtype=float)
    ng = np.asarray(ng_arr, dtype=float)
    out = {}
    out['min_xs'] = float(np.min(xs))
    out['max_xs'] = float(np.max(xs))
    out['min_ng'] = float(np.min(ng))
    out['max_ng'] = float(np.max(ng))
    diff = np.abs(xs - ng)
    out['max_abs_diff'] = float(np.max(diff))
    # relative diff, avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(np.abs(ng) > 0, diff / np.abs(ng), np.nan)
    out['max_rel_diff'] = float(np.nanmax(rel))
    return out


def _print_stats(label, stats):
    print(f"{label}")
    print(f"  Xsuite: min={stats['min_xs']:.6e}  max={stats['max_xs']:.6e}")
    print(f"  MAD-NG: min={stats['min_ng']:.6e}  max={stats['max_ng']:.6e}")
    print(f"  max|abs diff|={stats['max_abs_diff']:.6e}")
    print(f"  max|rel diff|={stats['max_rel_diff']:.6e}\n")


def _stats_row(name, stats):
    return [
        name,
        f"{stats['min_xs']:.6e}",
        f"{stats['min_ng']:.6e}",
        f"{stats['max_xs']:.6e}",
        f"{stats['max_ng']:.6e}",
        f"{stats['max_abs_diff']:.6e}",
        f"{stats['max_rel_diff']:.6e}",
    ]


def _print_table(title, rows):
    table = PrettyTable()
    table.field_names = [
        "quantity",
        "xs_min",
        "ng_min",
        "xs_max",
        "ng_max",
        "max_abs_diff",
        "max_rel_diff",
    ]
    for r in rows:
        table.add_row(r)
    print(f"\n{title}:\n{table}\n")

print(f"Loading lattice from: {LATTICE_PATH}")
line = xt.load(LATTICE_PATH)

# mat kick mat -> TKT
# drift kick drift exact -> DKD (exact)
# drift kick drift expanded -> error

# tt = line.get_table()
# tt_drifts = tt.rows[(tt.element_type == 'Drift')]
# tt_sext = tt.rows[(tt.element_type == 'Sextupole')]
# tt_quads = tt.rows[(tt.element_type == 'Quadrupole')]

# line.set(tt_drifts, model='exact')
# line.set(tt_sext, model='drift-kick-drift-exact')
# line.set(tt_quads, integrator='yoshida4')
for i in line.elements:
    if hasattr(i, 'rot_s_rad_no_frame'):
        i.rot_s_rad_no_frame = 0.0
    # if hasattr(i, 'delta_taper'):
    #     i.delta_taper = 0.0

line.configure_radiation(None, None, None)

mng = line.to_madng(sequence_name="lhcb1", temp_fname="020_fcc_temp",
                    debug=True, keep_files=True, redirect_stderr=True, stdout="020_log.txt")

# Before starting a twiss which might fail, track a particles through the lattice and save the orbit

p0 = line.build_particles()

p0_xs = p0.copy()
p0_ng = p0.copy()

X0 = [
        {
            'x': float(p0.x[i]),
            'px': float(p0.px[i]),
            'y': float(p0.y[i]),
            'py': float(p0.py[i]),
            't': float(p0.zeta[i] / p0.beta0[0]),
            'pt': float(p0.ptau[i]),
        } for i in range(len(p0.x))
    ]

mng['X0'] = X0

ng_script = """
    tbl = track {
        X0=X0,
        sequence=lhcb1,
        nturn=1,
        observe=0,
        save='atall',
    }
    """
mng.send(ng_script)

df = mng.tbl.to_df()
row = df[(df['slc'] == -2)]
coords = {coord: np.array(row[coord]) for coord in
    ['x', 'px', 'y', 'py', 't', 'pt']}

coords['zeta'] = coords.pop('t') * line.particle_ref.beta0[0]
coords['ptau'] = coords.pop('pt')

line.track(p0_xs, turn_by_turn_monitor="ONE_TURN_EBE")
mon = line.record_last_track
dct = mon.to_dict()
mon2 = xt.ParticlesMonitor.from_dict(dct)

diff_x = np.abs(mon2.x[0] - coords['x'][:-1])
diff_px = np.abs(mon2.px[0] - coords['px'][:-1])
diff_y = np.abs(mon2.y[0] - coords['y'][:-1])
diff_py = np.abs(mon2.py[0] - coords['py'][:-1])

# plot differences
fig, ax1 = plt.subplots()
ln1 = ax1.plot(mon2.s[0][:10000], diff_x[:10000], 'tab:blue', label='|x_xs - x_ng|')
ax1.set_xlabel('s [m]')
ax1.set_ylabel('Absolute x difference [m]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ln2 = ax2.plot(mon2.s[0][:10000], diff_px[:10000], 'tab:red', label='|px_xs - px_ng|')
ax2.set_ylabel('Absolute px difference', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.legend()
plt.tight_layout()

plt.show()

breakpoint()

# Xsuite track particle p0 and save position on every element


print("Computing CLOSED (periodic) Xsuite twiss (chrom=True)...")
tw_xs_closed = line.twiss(method='4d', chrom=True)

print("Computing CLOSED (periodic) MAD-NG twiss (compute_chromatic_properties=True, xsuite_tw=False) ...")
tw_ng_closed = line.madng_twiss(compute_chromatic_properties=True, xsuite_tw=False)

closed_pairs = [
    ("x", tw_xs_closed.x, tw_ng_closed.x_ng),
    ("px", tw_xs_closed.px, tw_ng_closed.px_ng),
    ("y", tw_xs_closed.y, tw_ng_closed.y_ng),
    ("py", tw_xs_closed.py, tw_ng_closed.py_ng),
    ("betx", tw_xs_closed.betx, tw_ng_closed.beta11_ng),
    ("bety", tw_xs_closed.bety, tw_ng_closed.beta22_ng),
    ("alfx", tw_xs_closed.alfx, tw_ng_closed.alfa11_ng),
    ("alfy", tw_xs_closed.alfy, tw_ng_closed.alfa22_ng),
    ("dx", tw_xs_closed.dx, tw_ng_closed.dx_ng),
    ("dy", tw_xs_closed.dy, tw_ng_closed.dy_ng),
    ("dpx", tw_xs_closed.dpx, tw_ng_closed.dpx_ng),
    ("dpy", tw_xs_closed.dpy, tw_ng_closed.dpy_ng),
    ("mux", tw_xs_closed.mux, tw_ng_closed.mu1_ng),
    ("muy", tw_xs_closed.muy, tw_ng_closed.mu2_ng),
    ("wx", tw_xs_closed.wx_chrom, tw_ng_closed.wx_ng),
    ("wy", tw_xs_closed.wy_chrom, tw_ng_closed.wy_ng),
    ("ax", tw_xs_closed.ax_chrom, tw_ng_closed.ax_ng),
    ("ay", tw_xs_closed.ay_chrom, tw_ng_closed.ay_ng),
    ("bx", tw_xs_closed.bx_chrom, tw_ng_closed.bx_ng),
    ("by", tw_xs_closed.by_chrom, tw_ng_closed.by_ng),
]

rows = []
for name, xs_v, ng_v in closed_pairs:
    stats = _compute_stats(xs_v, ng_v)
    rows.append(_stats_row(name, stats))
_print_table("Closed Twiss comparisons (Xsuite vs MAD-NG)", rows)

# --------------------------------------------------
# Non-periodic (open/range) twiss seeded from start
# --------------------------------------------------
# Computing non-periodic (open) twiss seeded at START element
print("Computing non-periodic (open) twiss seeded at START element...")
# Extract beta at start from closed Xsuite twiss (assume START is not an IP)
betx0 = float(tw_xs_closed['betx', START])
bety0 = float(tw_xs_closed['bety', START])
alfx0 = float(tw_xs_closed['alfx', START])
alfy0 = float(tw_xs_closed['alfy', START])
dx0 = float(tw_xs_closed['dx', START])
dpx0 = float(tw_xs_closed['dpx', START])

tw_xs_open = line.twiss(
    start=START,
    end=END,
    betx=betx0,
    bety=bety0,
    alfx=alfx0,
    alfy=alfy0,
    dx=dx0,
    dpx=dpx0,
    chrom=True,
    method='4d',
)

tw_ng_open = line.madng_twiss(
    start=START,
    end=END,
    beta11=betx0,
    beta22=bety0,
    alfa11=alfx0,
    alfa22=alfy0,
    dx=dx0,
    dpx=dpx0,
    compute_chromatic_properties=True,
    xsuite_tw=False,
)

open_pairs = closed_pairs  # same set of quantities, but arrays come from open twiss
names = [name for name, _, _ in open_pairs]
rows = []
for name in names:
    # Xsuite field mapping (chromatic quantities have '_chrom' suffix)
    if name in ('wx', 'wy', 'ax', 'ay', 'bx', 'by'):
        xs_field = name + '_chrom'
    else:
        xs_field = name

    if hasattr(tw_xs_open, xs_field):
        xs_v = getattr(tw_xs_open, xs_field)
    else:
        xs_v = tw_xs_open[xs_field]

    # MAD-NG field mapping
    ng_map = {
        'betx': 'beta11_ng', 'bety': 'beta22_ng', 'alfx': 'alfa11_ng', 'alfy': 'alfa22_ng',
        'mux': 'mu1_ng', 'muy': 'mu2_ng', 'dx': 'dx_ng', 'dy': 'dy_ng', 'dpx': 'dpx_ng', 'dpy': 'dpy_ng'
    }
    if name in ng_map:
        ng_field = ng_map[name]
    else:
        ng_field = name + '_ng'

    ng_v = getattr(tw_ng_open, ng_field)
    stats = _compute_stats(xs_v, ng_v)
    rows.append(_stats_row(name, stats))
_print_table("Non-periodic (range) Twiss comparisons (Xsuite vs MAD-NG)", rows)

print("\nProceeding to off-axis orbit check (unchanged)...")

print("\nOff-axis orbit check (x0 = 0.5 mm):")
idx_start = line.element_names.index(START)
idx_end = line.element_names.index(END) + 1
p = line.build_particles(x=X_OFF, px=0.0, y=0.0, py=0.0, delta=0.0)
line.track(p, ele_start=idx_start, ele_stop=idx_end, turn_by_turn_monitor="ONE_TURN_EBE")
mon = line.record_last_track
x_track_xs = mon.x[0, idx_start:idx_end + 1]
y_track_xs = mon.y[0, idx_start:idx_end + 1]
px_track_xs = mon.px[0, idx_start:idx_end + 1]
py_track_xs = mon.py[0, idx_start:idx_end + 1]

tw_ng_off = line.madng_twiss(
    start=START,
    end=END,
    beta11=tw_xs_closed["betx", START],
    beta22=tw_xs_closed["bety", START],
    alfa11=tw_xs_closed["alfx", START],
    alfa22=tw_xs_closed["alfy", START],
    dx=tw_xs_closed["dx", START],
    dpx=tw_xs_closed["dpx", START],
    x=X_OFF,
    px=0.0,
    y=0.0,
    py=0.0,
    compute_chromatic_properties=True,
    xsuite_tw=False,
)

orbit_pairs = [
    ("x_orbit", x_track_xs, tw_ng_off.x_ng),
    ("y_orbit", y_track_xs, tw_ng_off.y_ng),
    ("px_orbit", px_track_xs, tw_ng_off.px_ng),
    ("py_orbit", py_track_xs, tw_ng_off.py_ng),
]

orbit_rows = []
for name, xs_v, ng_v in orbit_pairs:
    stats = _compute_stats(xs_v, ng_v)
    orbit_rows.append(_stats_row(name, stats))
_print_table("Off-axis orbit comparisons (Xsuite tracking vs MAD-NG twiss)", orbit_rows)

diff_x = np.abs(tw_xs_open.x - tw_ng_open.x_ng)
diff_px = np.abs(tw_xs_open.px - tw_ng_open.px_ng)
diff_y = np.abs(tw_xs_open.y - tw_ng_open.y_ng)
diff_py = np.abs(tw_xs_open.py - tw_ng_open.py_ng)

fig, ax1 = plt.subplots()
ln1 = ax1.plot(tw_xs_open.s, diff_x, 'tab:blue', label='|x_xs - x_ng|')
ax1.set_xlabel('s [m]')
ax1.set_ylabel('Absolute x difference [m]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ln2 = ax2.plot(tw_xs_open.s, diff_px, 'tab:red', label='|px_xs - px_ng|')
ax2.set_ylabel('Absolute px difference', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# combined legend
lns = ln1 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper right')

ax1.set_title('Difference between Xsuite and MAD-NG Twiss (open range)')
ax1.grid()
fig.tight_layout()

fig, ax1 = plt.subplots()
ln1 = ax1.plot(tw_xs_open.s, diff_y, 'tab:blue', label='|y_xs - y_ng|')
ax1.set_xlabel('s [m]')
ax1.set_ylabel('Absolute y difference [m]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ln2 = ax2.plot(tw_xs_open.s, diff_py, 'tab:red', label='|py_xs - py_ng|')
ax2.set_ylabel('Absolute py difference', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# combined legend
lns = ln1 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper right')

ax1.set_title('Difference between Xsuite and MAD-NG Twiss (open range)')
ax1.grid()
fig.tight_layout()
plt.show()


print("\nDone.")