"""Generate reference trajectories for ETHZ track with Slippery Split environment.

This script generates two reference trajectories for the ETHZ track:
1. Oracle Reference: Uses the actual friction map (μ=0.3 for y<0, μ=1.0 for y>=0)
2. Naive Reference: Uses constant friction (μ=1.0 everywhere)

Updated to use ETHZ track (matching Phase 2) instead of simple rectangular track.
Friction map divides by y=0 (bottom/top split).
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from bayes_race.utils.friction import get_friction
from bayes_race.raceline.minimize_time_variable import solve as solve_variable
from bayes_race.raceline.minimize_time import solve as solve_constant
from bayes_race.tracks import ETHZ
from bayes_race.params import ORCA


def compute_distance(x, y):
    """Compute cumulative distance along path."""
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    return s


def plot_velocity_comparison(x_oracle, y_oracle, v_oracle, t_oracle,
                            x_naive, y_naive, v_naive, t_naive,
                            track,
                            save_path='bayes_race/raceline/src/velocity_comparison.png'):
    """Plot velocity vs distance comparison with ETHZ track visualization."""
    
    # Compute cumulative distance along path
    s_oracle = compute_distance(x_oracle, y_oracle)
    s_naive = compute_distance(x_naive, y_naive)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Velocity vs Distance
    ax1 = axes[0]
    ax1.plot(s_oracle, v_oracle, 'b-', label='Oracle (with friction map)', linewidth=2)
    ax1.plot(s_naive, v_naive, 'r--', label='Naive (μ=1.0 everywhere)', linewidth=2)
    
    # Highlight ice zone (y < -0.5) - need to find corresponding distance indices
    ice_mask_oracle = y_oracle < -0.5
    if np.any(ice_mask_oracle):
        s_ice_min = s_oracle[ice_mask_oracle].min()
        s_ice_max = s_oracle[ice_mask_oracle].max()
        ax1.axvspan(s_ice_min, s_ice_max, alpha=0.2, color='cyan', label='Ice Zone (y < -0.5, right in plot)')
    
    ax1.set_xlabel('Distance along path [m]', fontsize=12)
    ax1.set_ylabel('Velocity [m/s]', fontsize=12)
    ax1.set_title('ETHZ Track: Velocity Profile Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Track visualization with velocity color coding (matching plot_error_mpc_orca.py style)
    ax2 = axes[1]
    ax2.set_aspect('equal')
    
    # Plot track boundaries (using -y for x-axis to match existing style)
    ax2.plot(-track.y_outer, track.x_outer, 'k', lw=0.5, alpha=0.5)
    ax2.plot(-track.y_inner, track.x_inner, 'k', lw=0.5, alpha=0.5)
    
    # Plot raceline
    ax2.plot(-track.y_raceline, track.x_raceline, '--k', lw=0.5, alpha=0.3, label='Reference raceline')
    
    # Plot trajectories with velocity color coding
    scatter1 = ax2.scatter(-y_oracle, x_oracle, c=v_oracle, cmap='viridis', 
                          s=20, label='Oracle', alpha=0.7, zorder=3)
    scatter2 = ax2.scatter(-y_naive, x_naive, c=v_naive, cmap='plasma', 
                          s=20, marker='x', label='Naive', alpha=0.7, zorder=3)
    
    # Draw y=-0.5 line (vertical line in plot since x-axis is -y, y-axis is x)
    # In plot coordinates: x-axis is -y, y-axis is x
    # So y=-0.5 appears as a vertical line at x=0.5 in plot coordinates (since x-axis is -y)
    # Ice zone is y < -0.5, which is right side of plot (plotted x > 0.5)
    x_range = [min(track.x_raceline.min(), x_oracle.min(), x_naive.min()),
               max(track.x_raceline.max(), x_oracle.max(), x_naive.max())]
    y_range = [min(-track.y_raceline.max(), -y_oracle.max(), -y_naive.max()),
               max(-track.y_raceline.min(), -y_oracle.min(), -y_naive.min())]
    
    # Draw y=-0.5 boundary (vertical line in plot, since x-axis is -y)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
               label='y=-0.5 (Ice/Asphalt boundary, right=ice)', zorder=2)
    
    ax2.set_xlabel('$x$ [m] (plot: -y)', fontsize=12)
    ax2.set_ylabel('$y$ [m] (plot: x)', fontsize=12)
    ax2.set_title('ETHZ Track Visualization: Velocity Color Map', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.colorbar(scatter1, ax=ax2, label='Velocity [m/s]')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved velocity comparison plot to {save_path}")
    plt.close()


def generate_references():
    """Generate oracle and naive reference trajectories for ETHZ track."""
    
    print("=" * 60)
    print("ETHZ Track - Slippery Split Reference Generation (Phase 1)")
    print("=" * 60)
    
    # Create output directory
    src_dir = 'bayes_race/raceline/src'
    os.makedirs(src_dir, exist_ok=True)
    print(f"\n1. Output directory: {src_dir}")
    
    # Load ETHZ track
    print("\n2. Loading ETHZ track...")
    track = ETHZ(reference='optimal', longer=True)
    
    # Use raceline waypoints for optimization
    x, y = track.x_raceline, track.y_raceline
    print(f"   Track waypoints: {len(x)}")
    print(f"   Track bounds: x=[{x.min():.2f}, {x.max():.2f}], y=[{y.min():.2f}, {y.max():.2f}]")
    
    # Verify track crosses y=-0.5 (required for friction map)
    # Ice zone: y < -0.5 (right side of plot, since plot x-axis is -y)
    if y.min() >= -0.5 or y.max() <= -0.5:
        raise ValueError("ERROR: ETHZ track does not cross y=-0.5! Cannot apply friction map.")
    
    n_ice = np.sum(y < -0.5)
    n_asphalt = np.sum(y >= -0.5)
    print(f"   ✓ Track crosses y=-0.5: {n_ice} waypoints in ice zone (y<-0.5, right in plot), {n_asphalt} in asphalt zone (y>=-0.5, left in plot)")
    
    # Vehicle parameters
    print("\n3. Loading vehicle parameters...")
    params = ORCA()
    mass = params['mass']
    lf = params['lf']
    lr = params['lr']
    print(f"   Mass: {mass:.4f} kg, lf: {lf:.4f} m, lr: {lr:.4f} m")
    
    # Generate Oracle Reference (with friction map)
    print("\n4. Generating Oracle reference (with friction map)...")
    print("   This may take several minutes...")
    try:
        x_oracle, y_oracle, v_oracle, t_oracle, U_oracle, B_oracle = solve_variable(
            x, y, mass, lf, lr,
            friction_func=get_friction,
            plot_results=False,
            print_updates=True
        )
        print(f"   ✓ Oracle reference generated: lap time = {t_oracle:.4f} s")
    except Exception as e:
        print(f"   ✗ Error generating oracle reference: {e}")
        raise
    
    # Generate Naive Reference (constant μ=1.0)
    print("\n5. Generating Naive reference (constant μ=1.0)...")
    print("   This may take several minutes...")
    try:
        x_naive, y_naive, v_naive, t_naive, U_naive, B_naive = solve_constant(
            x, y, mass, lf, lr,
            plot_results=False,
            print_updates=True
        )
        print(f"   ✓ Naive reference generated: lap time = {t_naive:.4f} s")
    except Exception as e:
        print(f"   ✗ Error generating naive reference: {e}")
        raise
    
    # Save references
    print("\n6. Saving reference files...")
    oracle_path = os.path.join(src_dir, 'ethz_oracle_ref.npz')
    naive_path = os.path.join(src_dir, 'ethz_naive_ref.npz')
    
    np.savez(oracle_path,
             x=x_oracle, y=y_oracle, v=v_oracle, t=t_oracle,
             U=U_oracle, B=B_oracle)
    print(f"   ✓ Saved oracle reference to {oracle_path}")
    
    np.savez(naive_path,
             x=x_naive, y=y_naive, v=v_naive, t=t_naive,
             U=U_naive, B=B_naive)
    print(f"   ✓ Saved naive reference to {naive_path}")
    
    # Generate validation plot
    print("\n7. Generating validation plots...")
    plot_path = os.path.join(src_dir, 'ethz_velocity_comparison.png')
    plot_velocity_comparison(x_oracle, y_oracle, v_oracle, t_oracle,
                            x_naive, y_naive, v_naive, t_naive,
                            track,
                            save_path=plot_path)
    
    # Print validation statistics
    print("\n8. Validation Statistics:")
    print("-" * 60)
    
    ice_indices_oracle = np.where(y_oracle < -0.5)[0]
    ice_indices_naive = np.where(y_naive < -0.5)[0]
    
    if len(ice_indices_oracle) > 0 and len(ice_indices_naive) > 0:
        v_oracle_ice = v_oracle[ice_indices_oracle]
        v_naive_ice = v_naive[ice_indices_naive]
        
        v_oracle_asphalt = v_oracle[y_oracle >= -0.5]
        v_naive_asphalt = v_naive[y_naive >= -0.5]
        
        print(f"  Ice Zone (y < -0.5, right in plot):")
        print(f"    Oracle mean velocity: {v_oracle_ice.mean():.4f} m/s")
        print(f"    Naive mean velocity:  {v_naive_ice.mean():.4f} m/s")
        reduction = (1 - v_oracle_ice.mean() / v_naive_ice.mean()) * 100
        print(f"    Velocity reduction:    {reduction:.2f}%")
        
        print(f"\n  Asphalt Zone (y >= -0.5, left in plot):")
        print(f"    Oracle mean velocity: {v_oracle_asphalt.mean():.4f} m/s")
        print(f"    Naive mean velocity:  {v_naive_asphalt.mean():.4f} m/s")
        
        print(f"\n  Overall Lap Times:")
        print(f"    Oracle: {t_oracle:.4f} s")
        print(f"    Naive:  {t_naive:.4f} s")
        time_increase = ((t_oracle - t_naive) / t_naive) * 100
        print(f"    Time increase: {time_increase:.2f}%")
        
        # Validation check
        print(f"\n  Validation:")
        if reduction >= 20:
            print(f"    ✓ PASS: Velocity reduction ({reduction:.2f}%) >= 20%")
        else:
            print(f"    ⚠ WARNING: Velocity reduction ({reduction:.2f}%) < 20%")
    else:
        print("  ⚠ WARNING: Could not find ice zone (y < -0.5) in trajectories")
    
    print("\n" + "=" * 60)
    print("Phase 1 Reference generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    generate_references()
