from typing import Optional
import errno
import numpy as np
from numpy.typing import NDArray
from schema import DisplacementFrames

from MDAnalysis.coordinates.base import ReaderBase
from MDAnalysis.coordinates.timestep import Timestep


# Given that some displacements can be too small to easily see, it is
# sometimes best to scale them to magnify the visual effects. Atomic
# displacements will be scaled by the factor specified below before being
# added to the reference frame. Note that this scaling can be done
# dynamically while the simulation is running via either the
# `TrajectoryGenerator` or `TrajectoryPlayback` entities.


class SimpleGenerator(ReaderBase):
    """Trajectory entity for iterating over D-NEMD frames.

    This is designed to take the place of a standard MDAnalysis trajectory entity.
    Its primary task is to act as a data source from which MDAnalysis universe
    entities can poll molecular dynamics trajectory frames from.

    """

    def __init__(
            self, filename: str, confidence_level: int = 0,
            displacement_scale_factor: float = 1.0, **kwargs):
        """Trajectory entity for iterating over D-NEMD frames.

        This is designed to take the place of a standard MDAnalysis trajectory
        entity. Its primary task is to act as a data source from which MDAnalysis
        universe entities can poll molecular dynamics trajectory frames from.

        Arguments:
            filename: path to the file that is to be read.
            displacement_scale_factor: the factor by which displacements are to
                be scaled before adding them to the reference structure when
                generating a trajectory frame. This can be used to exaggerate
                atomic displacements that would otherwise be too small to easily
                see. Note that this only effects the trajectory positions stored
                in the `TimeStep` object. Default = 1.0.
            confidence_level: not yet implemented.

        """
        super().__init__(filename, **kwargs)

        self._displacement_frames = DisplacementFrames.read(
            filename, memory_mapped=True)

        self.confidence_level = confidence_level

        self._displacement_scale_factor = displacement_scale_factor

        # Developer's Note:
        # -----------------
        # Within the abstract base classes the `ts` attribute is assigned
        # dynamically within functions rather than during initialisation. This
        # design may lead to inconsistent behaviour and unclear expectations
        # regarding its availability, which can complicate debugging and
        # maintenance. It would be preferable for `ts` to be properly defined
        # within the initialiser to ensure predictable access. Hens, the time-
        # -step attribute is defined here:
        self.ts: Timestep = Timestep(
            self._displacement_frames.number_of_atoms_in_reference_structure)

        self._minimum_displacement_distance = None
        self._maximum_displacement_distance = None

    @property
    def n_frames(self):
        """Number of frames in the trajectory generator."""
        return self._displacement_frames.number_of_frames

    @property
    def displacement_scale_factor(self):
        """Factor to scale displacements by when added to the reference frame."""
        return self._displacement_scale_factor

    @displacement_scale_factor.setter
    def displacement_scale_factor(self, value: float):
        # Update the internal displacement scale factor attribute and then trigger
        # a re-read of the current frame to reconstruct the coordinates so that
        # they account for the newly updated scale factor.
        self._displacement_scale_factor = value
        self._read_frame(self.ts.frame)

    def _read_frame_into(self, frame: int, ts: Timestep) -> Timestep:
        """Read a specific frame.

        This method loads the data at the requested frame into the `Timestep`
        entity provided via the ``ts`` argument. If no time step entity is
        provided, then the instance's internal time step entity is used.

        Arguments:
            frame: index of the frame to be read.
            ts: optionally, a `Timestep` entity may be provided into which the
                frame data will be stored. If not specified then the frame data
                will be stored into the class instance's own time step entity.

        Returns:
              timestep: a `Timestep` entity containing the requested frame data.
        """
        # Store data into the supplied timestep object if provided, otherwise
        # use the class instance's internal timestep attribute.
        ts = self.ts if not ts else ts

        # Set the `frame` index attribute of the `Timestep` entity to the
        # specified frame number.
        ts.frame = frame

        # Ensure that the frame index is not out of range.
        if ts.frame >= self.n_frames:
            raise IOError(errno.EIO, "End of trajectory file reached")

        # Update the positions
        ts.positions = self._get_scaled_positions_at_frame(ts.frame)

        # The time must be updated manually as temporal distribution of D-NEMD
        # displacement frames is not linear.
        ts.time = self._displacement_frames.frame_times[ts.frame]

        # Return the timestep instance
        return ts

    def _read_frame(self, frame: int) -> Timestep:
        """Read the specified frame.

        This method will move the trajectory generator object to a specific frame
        in the simulation. The data will be loaded into the trajectory generator's
        time step entity, which will then be returned.

        Arguments:
            frame: index of the frame to be read.

        Returns:
            timestep: a `Timestep` entity containing the requested frame data.
        """

        return self._read_frame_into(frame, self.ts)

    def _read_next_timestep(self, ts: Optional[Timestep] = None) -> Timestep:
        """Read the next frame in the trajectory.

        This method loads the data at in the next frame into the `Timestep`
        entity provided via the ``ts`` argument. If no time step entity is
        provided, then the instance's internal time step entity is used.

        Arguments:
            ts: optionally, a `Timestep` entity may be provided into which the
                frame data will be stored. If not specified then the frame data
                will be stored into the class instance's own time step entity.

        Returns:
              timestep: a `Timestep` entity containing the next frame's data.
        """
        ts = self.ts if not ts else ts
        return self._read_frame_into(ts.frame + 1, ts)

    def _get_positions_at_frame(self, index: int) -> NDArray[float]:
        """Get the positions of the atoms at the specified frame.

        This method returns an array specifying the positions of the atoms at
        the requested frame accounting for displacements. The positions are
        computed by adding the displacement vectors to the reference structure.

        Arguments:
            index: index of the frame for which positions should be returned.
                If no index is provided, then the index of the current frame
                will be used.

        Returns:
            positions: positions of the, potentially displaced, atoms at the
                specified frame.

        Notes:
            This method does not include any contributions from the displacement
            scale factor `displacement_scale_factor`. For positions that included
            said scale factor use `_get_scaled_positions_at_frame`. Furthermore,
            displacement vectors that fall outside the confidence level, as
            specified by `confidence_level`, will be not contribute.

        """
        # If no index is specified, then use that of the current frame.
        index = index if index else self.ts.frame
        # Get the positions of the atoms in reference frame. Use a copy so that the
        # original data is not modified.
        positions = self._displacement_frames.reference_structure_positions[:].copy()
        # Fetch the atomic index data so that we know which atom each displacement
        # is associated with.
        atomic_indices = self._displacement_frames.atomic_indices[:]
        # Get the displacement data  (accounting for the desired confidence level)
        displacements = self.get_displacements(index)
        # Add the displacements to the reference structure
        positions[atomic_indices, ...] += displacements
        # Finally, return the position array.
        return positions

    def _get_scaled_positions_at_frame(self, index: Optional[int] = None):
        """Get the scaled positions of the atoms at the specified frame.

        This method returns an array specifying the positions of the atoms at
        the requested frame accounting for scaled displacements. The positions
        are computed by adding the scaled displacement vectors to the reference
        structure.

        Arguments:
            index: index of the frame for which positions should be returned.
                If no index is provided, then the index of the current frame
                will be used.

        Returns:
            positions: positions of the, potentially displaced, atoms at the
                specified frame.

        Notes:
            This method scales displacements by `displacement_scale_factor` before
            adding them to the reference positions. For unscaled displacement
            vectors please use `_get_positions_at_frame`. Furthermore,
            displacement vectors that fall outside the confidence level, as
            specified by `confidence_level`, will be not contribute.
        """
        # This method proceed in much the same manner as the `_get_positions_at_frame`
        # method except for the displacements being scaled by `displacement_scale_factor`
        # before being added to the reference position array.
        index = index if index else self.ts.frame
        positions = self._displacement_frames.reference_structure_positions[:].copy()
        atomic_indices = self._displacement_frames.atomic_indices[:]
        displacements = self.get_displacements(index)
        positions[atomic_indices, ...] += displacements * self.displacement_scale_factor
        return positions

    def get_displacements(self, index: Optional[int] = None):
        """Get the displacement vectors at the requested index.

        This method returns the displacement vectors for the tracked atoms at
        the specified frame.

        Arguments:
            index: index for the frame at which the displacements at to be
                returned. If no index is provided, then the index of the
                current frame will be used.

        Returns:
             displacement_vectors: the displacement vectors of the tracked atoms
                at the specified frame.

        Notes:
            Displacement vectors will only be returned for the tracked atoms.
            Furthermore, displacement vectors that fall outside the confidence
            level, as specified by `confidence_level`, will be zeroed out.
        """
        index = index if index else self.ts.frame
        displacements = self._displacement_frames.displacement_vectors[index, :, :].copy()
        mask = self._get_confidence_mask(index)
        displacements[~mask, ...] = 0.0
        return displacements

    def get_displacement_norms(self, index: Optional[int] = None):
        """Get the displacement vector norms at the requested index.

        This method returns the displacement vector norms for the tracked atoms
        at the specified frame.

        Arguments:
            index: index for the frame at which the displacements at to be
                returned. If no index is provided, then the index of the
                current frame will be used.

        Returns:
             displacement_vector_norms: the displacement vector norms of the
                tracked atoms at the specified frame.

        Notes:
            Displacement vector norms will only be returned for the tracked
            atoms. Furthermore, displacement vector norms that fall outside the
            confidence level, as specified by `confidence_level`, will be
            zeroed out.
        """
        index = index if index else self.ts.frame
        displacement_norms = self._displacement_frames.displacement_norms[index, :].copy()
        mask = self._get_confidence_mask(index)
        displacement_norms[~mask, ...] = 0.0
        return displacement_norms

    def _get_confidence_mask(self, index: Optional[int] = None,
                             confidence_level: int = 0) -> NDArray[bool]:
        """Mask specifying which displacements are significant.

        This will return an array of boolean values indicating if the displacements
        of the tracked atoms is deemed to be statistically significant.

        Argument:
            index: index for the frame for which the confidence mask is to be
                returned. If no index is provided, then the index of the
                current frame will be used.
            confidence_level: the confidence level to be used. If no value is
                specified, then it will fall back to that defined by the class
                attribute `confidence_level`. Three values are permitted:

                    - `0`: no masking is performed.
                    - `1`: standard deviation of one.
                    - `2`: standard deviation of two.

        Returns:
            mask: an array that is `True` for atoms whose displacements are deemed
                to be statistically significant, and `False` otherwise.

        Notes:
            This is currently a placeholder that should be implemented when possible.
        """
        index = index if index else self.ts.frame
        confidence_level = confidence_level if confidence_level else self.confidence_level

        # TODO: refactor this to add in a valid statistical significance check
        match confidence_level:
            case 0:
                return np.ones(
                    self._displacement_frames.number_of_displacements, dtype=bool)
            case 1:
                return np.where(
                    self._displacement_frames.displacement_norms[index] >
                    self._displacement_frames.standard_error_1_norms[index])[0]
            case 2:
                return np.where(
                    self._displacement_frames.displacement_norms[index] >
                    self._displacement_frames.standard_error_2_norms[index])[0]
            case _:
                raise ValueError(
                    f"Only confidence level values [0, 3] are valid, \"{confidence_level}\" provided")

    def _reopen(self):
        # The act of "reopening" a file does not exactly make much sense in
        # the context of an HDF5 file as it is not being iterated through
        # line by line like traditional fixed-format files might. Thus, this
        # method will just reset the trajectory to the first frame.
        self.ts.frame = 0

    @property
    def minimum_displacement_distance(self) -> float:
        """Minimum displacement distance found in the trajectory."""
        if not self._minimum_displacement_distance:
            self.__set_displacement_extrema()

        return self._minimum_displacement_distance

    @property
    def maximum_displacement_distance(self) -> float:
        """Maximum displacement distance found in the trajectory."""
        if not self._maximum_displacement_distance:
            self.__set_displacement_extrema()

        return self._maximum_displacement_distance

    def __set_displacement_extrema(self):
        """Identify and set the minimum and maximum displacement distances.

        This is an internal helper method that identities the minimum & maximum
        displacement distances and assigns them to their appropriate fields.
        """
        # Compute the size of the displacement norm array.
        array_size = (
                self._displacement_frames.number_of_frames
                * self._displacement_frames.number_of_displacements)

        # If the displacement array contains fewer than one hundred million values
        # then then load all data in from file and compute the extrema values directly.
        if array_size < 1_000_000_000:
            displacement_norms = self._displacement_frames.displacement_norms[:, :]
            min_value, max_value = np.min(displacement_norms), np.max(displacement_norms)
            del displacement_norms

        # However, if there is a significant amount of data then just load the
        # data piece by piece and compute the extrema values via a loop. This
        # is much slower and less efficient in Python but a safety catch is
        # needed for situations where large amounts of data are involved.
        else:
            max_value = -9999999999
            min_value = +9999999999

            for i in range(self._displacement_frames.number_of_frames):

                displacement_norms = self._displacement_frames.displacement_norms[i, ...]
                frame_min, frame_max = np.min(displacement_norms), np.max(displacement_norms)

                if frame_max > max_value:
                    max_value = frame_max

                if frame_min < min_value:
                    min_value = frame_min

        self._minimum_displacement_distance = min_value
        self._maximum_displacement_distance = max_value
