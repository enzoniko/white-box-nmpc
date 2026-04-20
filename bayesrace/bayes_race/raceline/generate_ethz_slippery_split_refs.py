"""Generate reference trajectories for ETHZ track with Slippery Split environment.

This script generates two reference trajectories for the ETHZ track:
1. Oracle Reference: Uses the actual friction map (μ=0.3 for x<0, μ=1.0 for x>=0)
2. Naive Reference: Uses constant friction (μ=1.0 everywhere)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayes_race.utils.friction import get_friction
from bayes_race.raceline.minimize_time_variable import solve as solve_variable
from bayes_race.raceline.minimize_time import solve as solve_constant
from bayes_race.tracks import ETHZ
from bayes_race.params import ORCA


def generate_references():
    """Generate oracle and naive reference trajectories for ETHZ track."""
    
    print("=" * 60)
    print("ETHZ Track - Slippery Split Reference Generation")
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
    
    # Check if track crosses x=0
    if x.min() >= 0 or x.max() <= 0:
        print("   ⚠ WARNING: ETHZ track does not cross x=0!")
        print("   Friction map will still be applied, but ice zone may be limited")
    else:
        n_ice = np.sum(x < 0)
        n_asphalt = np.sum(x >= 0)
        print(f"   ✓ Track crosses x=0: {n_ice} waypoints in ice zone, {n_asphalt} in asphalt zone")
    
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
    plot_velocity_comparison(x_oracle, y_oracle, v_oracle, t_oracle,
                            x_naive, y_naive, v_naive, t_naive,
                            save_path=os.path.join(src_dir, 'ethz_velocity_comparison.png'))
    
    # Print validation statistics
    print("\n8. Validation Statistics:")
    print("-" * 60)
    
    ice_indices_oracle = np.where(x_oracle < 0)[0]
    ice_indices_naive = np.where(x_naive < 0)[0]
    
    if len(ice_indices_oracle) > 0 and len(ice_indices_naive) > 0:
        v_oracle_ice = v_oracle[ice_indices_oracle]
        v_naive_ice = v_naive[ice_indices_naive]
        
        print(f"  Ice Zone (x < 0):")
        print(f"    Oracle mean velocity: {v_oracle_ice.mean():.4f} m/s")
        print(f"    Naive mean velocity:  {v_naive_ice.mean():.4f} m/s")
        reduction = (1 - v_oracle_ice.mean() / v_naive_ice.mean()) * 100
        print(f"    Velocity reduction:    {reduction:.2f}%")
        
        print(f"\n  Overall Lap Times:")
        print(f"    Oracle: {t_oracle:.4f} s")
        print(f"    Naive:  {t_naive:.4f} s")
        time_increase = ((t_oracle - t_naive) / t_naive) * 100
        print(f"    Time increase: {time_increase:.2f}%")
        
        if reduction >= 20:
            print(f"\n  ✓ PASS: Velocity reduction ({reduction:.2f}%) >= 20%")
        else:
            print(f"\n  ⚠ WARNING: Velocity reduction ({reduction:.2f}%) < 20%")
    else:
        print("  ⚠ WARNING: Could not find ice zone (x < 0) in trajectories")
    
    print("\n" + "=" * 60)
    print("Reference generation complete!")
    print("=" * 60)


def compute_distance(x, y):
    """Compute cumulative distance along path."""
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    return s


def plot_velocity_comparison(x_oracle, y_oracle, v_oracle, t_oracle,
                            x_naive, y_naive, v_naive, t_naive,
                            save_path='ethz_velocity_comparison.png'):
    """Plot velocity vs distance comparison."""
    
    s_oracle = compute_distance(x_oracle, y_oracle)
    s_naive = compute_distance(x_naive, y_naive)
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_oracle, v_oracle, 'b-', label='Oracle (with friction map)', linewidth=2)
    plt.plot(s_naive, v_naive, 'r--', label='Naive (μ=1.0 everywhere)', linewidth=2)
    
    ice_mask_oracle = x_oracle < 0
    if np.any(ice_mask_oracle):
        s_ice_min = s_oracle[ice_mask_oracle].min()
        s_ice_max = s_oracle[ice_mask_oracle].max()
        plt.axvspan(s_ice_min, s_ice_max, alpha=0.2, color='cyan', label='Ice Zone (x < 0)')
    
    plt.xlabel('Distance along path [m]')
    plt.ylabel('Velocity [m/s]')
    plt.title('ETHZ Track: Velocity Profile Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved velocity comparison plot to {save_path}")
    plt.close()


if __name__ == '__main__':
    generate_references()
