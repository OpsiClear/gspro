"""
Example: Gaussian splat filtering usage.

Demonstrates how to use the gspro filtering system for:
- Volume filtering (sphere and cuboid)
- Opacity filtering
- Scale filtering
- Combined filtering
"""

import logging

import numpy as np

from gspro.filter import (
    apply_filter,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
    filter_gaussians,
)

# Configure logging to see filtering statistics
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_sample_data(n: int = 10000):
    """Generate sample Gaussian splat data for demonstration."""
    np.random.seed(42)

    positions = np.random.randn(n, 3).astype(np.float32) * 2.0
    quaternions = np.random.randn(n, 4).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = np.random.rand(n, 3).astype(np.float32) * 2.0
    opacities = np.random.rand(n).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)

    # Add some outliers
    outlier_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    scales[outlier_idx] = 10.0  # Large outlier scales

    return {
        "positions": positions,
        "quaternions": quaternions,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }


def example_1_opacity_filtering():
    """Example 1: Simple opacity filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Opacity Filtering (kwargs like transform module)")
    print("=" * 70)

    # Generate data
    data = generate_sample_data()
    positions = data["positions"]
    opacities = data["opacities"]

    print(f"Original: {len(positions)} Gaussians")

    # Remove Gaussians with opacity < 10% (kwargs like transform module)
    mask = apply_filter(positions, opacities=opacities, opacity_threshold=0.1)

    filtered_positions = positions[mask]
    opacities[mask]

    print(f"After opacity filter: {len(filtered_positions)} Gaussians")
    print(f"Removed: {len(positions) - len(filtered_positions)} Gaussians")
    print(f"Kept: {mask.sum() / len(mask) * 100:.1f}%")


def example_2_scale_filtering():
    """Example 2: Scale filtering with auto-threshold."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Scale Filtering (Auto-Threshold, kwargs)")
    print("=" * 70)

    # Generate data
    data = generate_sample_data()
    positions = data["positions"]
    scales = data["scales"]

    print(f"Original: {len(positions)} Gaussians")

    # Calculate recommended threshold (99.5th percentile)
    recommended_threshold = calculate_recommended_max_scale(scales)
    print(f"Recommended max_scale threshold: {recommended_threshold:.4f}")

    # Apply scale filtering (kwargs like transform module)
    mask = apply_filter(positions, scales=scales, max_scale=recommended_threshold)

    filtered_positions = positions[mask]

    print(f"After scale filter: {len(filtered_positions)} Gaussians")
    print(f"Removed: {len(positions) - len(filtered_positions)} outliers")


def example_3_sphere_filtering():
    """Example 3: Sphere volume filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Sphere Volume Filtering (kwargs)")
    print("=" * 70)

    # Generate data
    data = generate_sample_data()
    positions = data["positions"]

    print(f"Original: {len(positions)} Gaussians")

    # Calculate scene bounds
    bounds = calculate_scene_bounds(positions)
    print(f"Scene bounds: min={bounds.min}, max={bounds.max}")
    print(f"Scene center: {bounds.center}")
    print(f"Scene max dimension: {bounds.max_size:.3f}")

    # Keep only Gaussians within 50% of scene radius from center (kwargs like transform)
    mask = apply_filter(
        positions,
        filter_type="sphere",
        sphere_center=tuple(bounds.center),
        sphere_radius_factor=0.5,
        scene_bounds=bounds,
    )

    filtered_positions = positions[mask]

    print(f"After sphere filter (50% radius): {len(filtered_positions)} Gaussians")
    print(f"Kept: {mask.sum() / len(mask) * 100:.1f}%")


def example_4_cuboid_filtering():
    """Example 4: Cuboid volume filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Cuboid Volume Filtering (kwargs)")
    print("=" * 70)

    # Generate data
    data = generate_sample_data()
    positions = data["positions"]

    print(f"Original: {len(positions)} Gaussians")

    # Calculate scene bounds
    bounds = calculate_scene_bounds(positions)

    # Keep only central 30% region (box) (kwargs like transform)
    mask = apply_filter(
        positions,
        filter_type="cuboid",
        cuboid_center=tuple(bounds.center),
        cuboid_size_factor_x=0.3,
        cuboid_size_factor_y=0.3,
        cuboid_size_factor_z=0.3,
        scene_bounds=bounds,
    )

    filtered_positions = positions[mask]

    print(f"After cuboid filter (30% size): {len(filtered_positions)} Gaussians")
    print(f"Kept: {mask.sum() / len(mask) * 100:.1f}%")


def example_5_combined_filtering():
    """Example 5: Combined filtering (all filters)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Combined Filtering (Volume + Opacity + Scale, tuple return)")
    print("=" * 70)

    # Generate data
    data = generate_sample_data()

    print(f"Original: {len(data['positions'])} Gaussians")

    # Calculate scene bounds and recommended scale threshold
    bounds = calculate_scene_bounds(data["positions"])
    recommended_scale = calculate_recommended_max_scale(data["scales"])

    # Combine all three filters (kwargs like transform, returns tuple like transform)
    new_pos, new_quats, new_scales, new_opac, new_colors = filter_gaussians(
        positions=data["positions"],
        quaternions=data["quaternions"],
        scales=data["scales"],
        opacities=data["opacities"],
        colors=data["colors"],
        filter_type="sphere",
        sphere_center=tuple(bounds.center),
        sphere_radius_factor=0.8,  # 80% of scene radius
        opacity_threshold=0.05,  # Remove < 5% opacity
        max_scale=recommended_scale,  # Remove outliers
        scene_bounds=bounds,
    )

    print(f"After combined filtering: {len(new_pos)} Gaussians")
    print(f"Removed: {len(data['positions']) - len(new_pos)} Gaussians")
    print(f"Kept: {len(new_pos) / len(data['positions']) * 100:.1f}%")

    # All attributes are filtered consistently (tuple return like transform)
    print("\nFiltered attributes shapes:")
    print(f"  positions: {new_pos.shape}")
    print(f"  quaternions: {new_quats.shape}")
    print(f"  scales: {new_scales.shape}")
    print(f"  opacities: {new_opac.shape}")
    print(f"  colors: {new_colors.shape}")


def example_6_reuse_bounds():
    """Example 6: Reusing scene bounds for multiple frames."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Reusing Scene Bounds (Multi-Frame, kwargs)")
    print("=" * 70)

    # Simulate multiple frames
    n_frames = 5

    # Calculate bounds from first frame
    first_frame = generate_sample_data()
    bounds = calculate_scene_bounds(first_frame["positions"])

    print("Calculated scene bounds from frame 0")
    print(f"Processing {n_frames} frames...")

    # Reuse bounds for all frames (kwargs like transform)
    for frame_idx in range(n_frames):
        # Generate frame data (in real use, load from file)
        frame_data = generate_sample_data()

        # Apply filtering with pre-calculated bounds (kwargs like transform)
        mask = apply_filter(
            frame_data["positions"],
            filter_type="sphere",
            sphere_radius_factor=0.7,
            scene_bounds=bounds,  # Reuse!
        )

        print(f"  Frame {frame_idx}: {len(frame_data['positions'])} -> {mask.sum()} Gaussians")


def example_7_custom_percentiles():
    """Example 7: Custom percentile for scale threshold."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Custom Percentiles for Scale Threshold (kwargs)")
    print("=" * 70)

    # Generate data
    data = generate_sample_data()
    scales = data["scales"]

    # Try different percentiles
    percentiles = [90.0, 95.0, 99.0, 99.5, 99.9]

    print("Scale thresholds at different percentiles:")
    for p in percentiles:
        threshold = calculate_recommended_max_scale(scales, percentile=p)
        print(f"  {p:5.1f}th percentile: {threshold:.4f}")

    # Use more aggressive filtering (95th percentile) (kwargs like transform)
    aggressive_threshold = calculate_recommended_max_scale(scales, percentile=95.0)
    mask = apply_filter(data["positions"], scales=scales, max_scale=aggressive_threshold)

    print("\nWith 95th percentile threshold:")
    print(f"  Kept: {mask.sum()} / {len(mask)} Gaussians ({mask.sum() / len(mask) * 100:.1f}%)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GSPRO FILTERING SYSTEM EXAMPLES")
    print("=" * 70)

    example_1_opacity_filtering()
    example_2_scale_filtering()
    example_3_sphere_filtering()
    example_4_cuboid_filtering()
    example_5_combined_filtering()
    example_6_reuse_bounds()
    example_7_custom_percentiles()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
