import numpy as np
from os.path import join
import re
import glob
from numpy.typing import NDArray
import errno
import MDAnalysis
from MDAnalysis.coordinates.timestep import Timestep
from MDAnalysis.coordinates.base import ProtoReader
from parser import NemdDisplacementFrame


class TrajectoryGenerator(ProtoReader):
    """A trajectory entity for iterating over NEMD frames.

    This is designed to take the place of a standard MDAnalysis trajectory entity.
    Its primary task is to act as a data source from which the MDAnalysis Universe
    entity can poll MD trajectory frames from.


    Warning:
        It is highly recommended that you do not modify this class unless you
        have a comprehensive understanding of its functionality. This class is
        complex and potentially hazardous. It has been assembled in an
        unconventional manner, using elements from the somewhat unconventional
        `ProtoReader` class provided by the the `MDAnalysis` framework. This
        implementation is minimally functional and potentially problematic. Use
        with caution. Note that comments and docstrings have been omitted until
        the class is cleaned up properly.

    Todo:
        1. Check if the `timestep._unitcell = self.dimensions_array[self.ts.frame]`
            call is really needed in the `._read_next_timestep` method.
        2. Add documentation and comments to the class.
        3. Do not commit.
        5. Refactor the class to be less.... poorly written?
    """
    def __init__(
            self, reference_structure: NDArray[float],
            displacements: NDArray[float],
            displacement_indices: NDArray[int],
            displacement_scale_factor: float = 1.0):
        """Initialise the TrajectoryGenerator class.

        Arguments:
            reference_structure: An Nx3 numpy array representing the reference
                structure, where "N" is the number of atoms.
            displacements: An FxCx3 numpy array containing displacement data,
                where "F" is the number of frames and C is the number of carbon-
                alpha atoms.
            displacement_indices: A numpy array of indices where displacements occur.
            displacement_scale_factor: A scale factor for the displacements. Defaults to 1.0.
        """


        super().__init__()

        self._reference_structure = reference_structure
        self._displacements = displacements
        self._displacement_indices = displacement_indices
        self._displacement_scale_factor = displacement_scale_factor

        self.n_atoms = reference_structure.shape[0]
        self.n_frames = displacements.shape[0]

        self.timestep = Timestep(self.n_atoms)

        # The time step delta time variable "dt" is set to 1ps to prevent
        # MDAnalysis from incessantly wining that no "dt" value has been
        # provided. The delta time value for NEMD is not necessarily a
        # consistent or straightforward thing.
        self.timestep.dt = 1.0

    @property
    def displacement_scale_factor(self):
        return self._displacement_scale_factor

    @displacement_scale_factor.setter
    def displacement_scale_factor(self, value: float):
        self._displacement_scale_factor = value

        # Trigger, a re-read of the current frame to reconstruct the
        # coordinates so that they account for the new scale factor.
        self.ts.frame = self.ts.frame - 1
        self._read_next_timestep()

    @property
    def ts(self):
        return self.timestep

    @ts.setter
    def ts(self, new_timestep: Timestep):
        self.timestep = new_timestep

    def __generate_trajectory_frame(self, frame_number):
        frame = self._reference_structure.copy()
        displacement = self._displacements[frame_number, ...] * self._displacement_scale_factor
        frame[self._displacement_indices] += displacement
        return frame

    @classmethod
    def from_nemd_displacement_frames(
            cls, frames: list[NemdDisplacementFrame],
            universe: MDAnalysis.Universe,
            confidence_level: int = 0,
            displacement_scale_factor: float = 1.0
    ) -> 'TrajectoryGenerator':

        # Check that the number of displacements is constant through the
        # displacement frames.
        atom_counts = [len(i.carbon_alpha_index) for i in frames]
        n_atoms = atom_counts[0]
        if not all(i == n_atoms for i in atom_counts):
            raise ValueError(
                "All `NemdDisplacementFrame` instances must have a constant "
                "number of displacements, but varying counts were "
                "encountered.")

        # Ensure that the atoms to which the displacements correspond to do
        # not change from one frame to the other.
        carbon_alpha_idx = frames[0].carbon_alpha_index
        if not all((i.carbon_alpha_index == carbon_alpha_idx).all() for i in frames):
            raise ValueError(
                "All `NemdDisplacementFrame` instances must have matching "
                "alpha index arrays, but varying arrays were "
                "encountered.")

        # Construct an index array map that translates "carbon-alpha" indices,
        # as specified in the displacement file, to atomic indices as found in
        # the structure file.
        carbon_alpha_index_to_atomic_index = np.where(
            universe.atoms.names == "CA")[0]

        n_frames = len(frames)
        displacements = np.zeros((n_frames, n_atoms, 3))

        confidence_attribute = [
            # Invalid attribute to trigger default return ┐
            "standard_error_of_norm_std0",  # <───────────┘
            'standard_error_of_norm_std1',
            "standard_error_of_norm_std2"][confidence_level]

        for i, frame in enumerate(frames):

            # Identify which displacements are statistically significant.
            threshold = getattr(frame, confidence_attribute, 0)
            source_indices = np.where(frame.average_displacement_norm > threshold)[0]

            # Use array indexing to assign displacements in one operation
            displacements[i, source_indices] = frame.average_displacement[source_indices]

        # Array specifying the index of the atom with which each displacement is
        # associated. This is used to apply the displacements during frame
        # generation.
        displacement_indices = carbon_alpha_index_to_atomic_index[frames[0].carbon_alpha_index - 1]

        return cls(universe.atoms.positions.copy(),
                   displacements, displacement_indices,
                   displacement_scale_factor=displacement_scale_factor)

    def _read_next_timestep(self, timestep=None):

        timestep = self.timestep if timestep is None else timestep

        if timestep.frame >= self.n_frames:
            raise IOError(errno.EIO, "End of trajectory file reached")

        new_coordinates = self.__generate_trajectory_frame(timestep.frame + 1)
        timestep.has_positions = True
        timestep._pos = new_coordinates

        timestep.frame += 1


        # timestep._unitcell = self.dimensions_array[self.ts.frame]
        timestep.time = timestep.frame * self.dt

        return timestep

    def _read_frame(self, i):
        """read frame i"""
        self.ts.frame = i - 1
        return self._read_next_timestep()

    def _reopen(self):
        """Reset iteration to first frame"""
        self.ts.frame = -1
        self.ts.time = -1


def auto_load_displacement_frames(path: str) -> list[NemdDisplacementFrame]:
    """Convenience function for parsing loading of displacement frame files.

    When provided with a path to a directory, this function will locate all files
    of the form "average_xyz_displacement_*ps". Matching files will be parsed
    into a list of `NemdDisplacementFrame` entities. Note that displacement frames
    will be sorted in chronological order according to the timestep specified
    in their file names.

    Arguments:
        path: the path to the directory within which the displacement data files
            are all stored.

    Returns:
        displacement_frames: a chronologically sorted list of displacement frames.
    """
    # Locate all files matching the expression "average_xyz_displacement_*ps"
    # where "*" is the time elapsed.
    files = glob.glob(join(path, "average_xyz_displacement_*ps"))

    # Sort the file list so that frames will be in chronological order
    pattern = re.compile(r'(?<=average_xyz_displacement_)\d*(?=ps)')
    files.sort(key=lambda i: int(pattern.search(i).group(0)))

    # Parse the files into displacement frame entities
    displacement_frames = list(map(NemdDisplacementFrame.load, files))

    # Return the displacement frame list
    return displacement_frames

