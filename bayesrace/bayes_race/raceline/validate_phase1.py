"""Quick validation script for Phase 1 implementation."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from bayes_race.utils.friction import get_friction
from bayes_race.raceline.minimize_time_variable import solve as solve_variable
from bayes_race.raceline.minimize_time import solve as solve_constant
from bayes_race.params import ORCA


def create_simple_track():
    """Create a simple track that crosses x=0."""
    # Simple straight line that crosses x=0
    n = 50  # Fewer waypoints for faster optimization
    x = np.linspace(-10, 10, n)
    y = np.zeros(n)
    return x, y


def main():
    print("Phase 1 Validation - Quick Test")
    print("=" * 50)
    
    # Create track
    x, y = create_simple_track()
    print(f"Track created: {len(x)} waypoints")
    print(f"  x-range: [{x.min():.2f}, {x.max():.2f}]")
    
    # Vehicle params
    params = ORCA()
    mass = params['mass']
    lf = params['lf']
    lr = params['lr']
    
    # Test oracle (variable friction)
    print("\n1. Testing Oracle reference (variable friction)...")
    try:
        x_oracle, y_oracle, v_oracle, t_oracle, U_oracle, B_oracle = solve_variable(
            x, y, mass, lf, lr,
            friction_func=get_friction,
            plot_results=False,
            print_updates=False
        )
        print(f"   ✓ Success: lap time = {t_oracle:.4f} s")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Test naive (constant friction)
    print("\n2. Testing Naive reference (constant friction)...")
    try:
        x_naive, y_naive, v_naive, t_naive, U_naive, B_naive = solve_constant(
            x, y, mass, lf, lr,
            plot_results=False,
            print_updates=False
        )
        print(f"   ✓ Success: lap time = {t_naive:.4f} s")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Validation
    print("\n3. Validation:")
    ice_mask_oracle = x_oracle < 0
    ice_mask_naive = x_naive < 0
    
    if np.any(ice_mask_oracle) and np.any(ice_mask_naive):
        v_oracle_ice = v_oracle[ice_mask_oracle].mean()
        v_naive_ice = v_naive[ice_mask_naive].mean()
        reduction = (1 - v_oracle_ice / v_naive_ice) * 100
        
        print(f"   Ice zone mean velocity:")
        print(f"     Oracle: {v_oracle_ice:.4f} m/s")
        print(f"     Naive:  {v_naive_ice:.4f} m/s")
        print(f"     Reduction: {reduction:.2f}%")
        
        if reduction >= 20:
            print(f"   ✓ PASS: Velocity reduction >= 20%")
        else:
            print(f"   ⚠ WARNING: Velocity reduction < 20%")
    
    # Save results
    src_dir = 'src'
    os.makedirs(src_dir, exist_ok=True)
    
    np.savez(f'{src_dir}/oracle_ref.npz',
             x=x_oracle, y=y_oracle, v=v_oracle, t=t_oracle,
             U=U_oracle, B=B_oracle)
    np.savez(f'{src_dir}/naive_ref.npz',
             x=x_naive, y=y_naive, v=v_naive, t=t_naive,
             U=U_naive, B=B_naive)
    
    print(f"\n4. Files saved to {src_dir}/")
    print("   ✓ oracle_ref.npz")
    print("   ✓ naive_ref.npz")
    
    # Simple plot
    s_oracle = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_oracle)**2 + np.diff(y_oracle)**2))])
    s_naive = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_naive)**2 + np.diff(y_naive)**2))])
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_oracle, v_oracle, 'b-', label='Oracle (with friction map)', linewidth=2)
    plt.plot(s_naive, v_naive, 'r--', label='Naive (μ=1.0 everywhere)', linewidth=2)
    
    if np.any(ice_mask_oracle):
        s_ice_min = s_oracle[ice_mask_oracle].min()
        s_ice_max = s_oracle[ice_mask_oracle].max()
        plt.axvspan(s_ice_min, s_ice_max, alpha=0.2, color='cyan', label='Ice Zone (x < 0)')
    
    plt.xlabel('Distance along path [m]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity Profile Comparison: Oracle vs Naive')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{src_dir}/velocity_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ velocity_comparison.png")
    
    print("\n" + "=" * 50)
    print("Validation complete!")


if __name__ == '__main__':
    main()
