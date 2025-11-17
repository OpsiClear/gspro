"""Convenience API for filter mask layer management using GSData.masks."""

from gsply import GSData

from gspro.filter.pipeline import Filter


class FilterMasks:
    """Manage named filter mask layers in GSData.

    This class provides a convenience wrapper around GSData's mask layer functionality,
    specifically designed for Filter operations.

    Attributes:
        data: The GSData object whose mask layers are managed

    Example:
        >>> from gsply import GSData
        >>> from gspro import Filter, FilterMasks
        >>>
        >>> # Create data
        >>> data = plyread("scene.ply")
        >>>
        >>> # Add filter mask layers
        >>> masks = FilterMasks(data)
        >>> masks.add("opacity", Filter().min_opacity(0.3))
        >>> masks.add("sphere", Filter().within_sphere(radius=0.8))
        >>> masks.add("scale", Filter().max_scale(2.0))
        >>>
        >>> # Inspect layers
        >>> masks.summary()
        >>> # Output:
        >>> # opacity: 71/100 (71.0%)
        >>> # sphere: 91/100 (91.0%)
        >>> # scale: 88/100 (88.0%)
        >>>
        >>> # Apply filters
        >>> filtered = masks.apply(mode="and")  # All layers must pass
        >>> filtered = masks.apply(layers=["opacity", "sphere"])  # Specific layers
    """

    def __init__(self, data: GSData):
        """Initialize FilterMasks for a GSData object.

        Args:
            data: GSData object to manage mask layers for
        """
        self.data = data

    def add(self, name: str, filter: Filter) -> None:
        """Add a named filter mask layer.

        Args:
            name: Name for this mask layer
            filter: Filter pipeline to apply

        Example:
            >>> masks.add("high_opacity", Filter().min_opacity(0.5))
            >>> masks.add("foreground", Filter().within_sphere(radius=0.8))
        """
        mask = filter.get_mask(self.data)
        self.data.add_mask_layer(name, mask)

    def remove(self, name: str) -> None:
        """Remove a mask layer by name.

        Args:
            name: Name of the mask layer to remove

        Example:
            >>> masks.remove("scale")
        """
        self.data.remove_mask_layer(name)

    def get(self, name: str):
        """Get a mask layer by name.

        Args:
            name: Name of the mask layer

        Returns:
            Boolean mask array of shape (N,)

        Example:
            >>> opacity_mask = masks.get("opacity")
            >>> print(f"{opacity_mask.sum()} Gaussians pass opacity filter")
        """
        return self.data.get_mask_layer(name)

    def apply(
        self, mode: str = "and", layers: list[str] | None = None, inplace: bool = False
    ) -> GSData:
        """Apply mask layers to filter Gaussians.

        Args:
            mode: Combination mode - "and" (all must pass) or "or" (any must pass)
            layers: List of layer names to apply (None = use all layers)
            inplace: If True, modify data in-place; if False, return filtered copy

        Returns:
            Filtered GSData (self.data if inplace=True, new GSData if inplace=False)

        Example:
            >>> # Filter using all layers (AND logic)
            >>> filtered = masks.apply(mode="and")
            >>>
            >>> # Filter using specific layers (OR logic)
            >>> filtered = masks.apply(mode="or", layers=["opacity", "scale"])
            >>>
            >>> # Filter in-place
            >>> masks.apply(mode="and", inplace=True)
        """
        return self.data.apply_masks(mode=mode, layers=layers, inplace=inplace)

    def combine(self, mode: str = "and", layers: list[str] | None = None):
        """Combine mask layers into a single boolean mask.

        Args:
            mode: Combination mode - "and" or "or"
            layers: List of layer names to combine (None = use all layers)

        Returns:
            Combined boolean mask of shape (N,)

        Example:
            >>> # Get combined mask for manual use
            >>> mask = masks.combine(mode="and")
            >>> print(f"{mask.sum()} Gaussians pass all filters")
            >>> custom_filtered = data[mask]
        """
        return self.data.combine_masks(mode=mode, layers=layers)

    def summary(self) -> None:
        """Print summary of all mask layers.

        Example:
            >>> masks.summary()
            >>> # Output:
            >>> # opacity: 71/100 (71.0%)
            >>> # sphere: 91/100 (91.0%)
            >>> # scale: 88/100 (88.0%)
        """
        if self.data.mask_names is None or self.data.masks is None:
            print("No mask layers")
            return

        n_total = len(self.data)
        for name in self.data.mask_names:
            mask = self.data.get_mask_layer(name)
            n_pass = mask.sum()
            pct = (n_pass / n_total * 100) if n_total > 0 else 0
            print(f"{name}: {n_pass}/{n_total} ({pct:.1f}%)")

    @property
    def names(self) -> list[str] | None:
        """Get list of mask layer names.

        Returns:
            List of mask layer names, or None if no layers exist

        Example:
            >>> print(f"Active layers: {masks.names}")
            >>> # Output: Active layers: ['opacity', 'sphere', 'scale']
        """
        return self.data.mask_names

    def __len__(self) -> int:
        """Return number of mask layers.

        Example:
            >>> print(f"{len(masks)} mask layers defined")
        """
        return len(self.data.mask_names) if self.data.mask_names is not None else 0

    def __contains__(self, name: str) -> bool:
        """Check if a mask layer exists.

        Example:
            >>> if "opacity" in masks:
            ...     print("Opacity filter is active")
        """
        return self.data.mask_names is not None and name in self.data.mask_names

    def __getitem__(self, name: str):
        """Get a mask layer by name (same as get()).

        Example:
            >>> opacity_mask = masks["opacity"]
        """
        return self.data.get_mask_layer(name)

    def __repr__(self) -> str:
        """Return string representation.

        Example:
            >>> print(masks)
            >>> # Output: FilterMasks(3 layers: opacity, sphere, scale)
        """
        if self.data.mask_names is None:
            return "FilterMasks(0 layers)"
        names_str = ", ".join(self.data.mask_names)
        return f"FilterMasks({len(self)} layers: {names_str})"
