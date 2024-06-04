from os.path import join
import re
import glob

import warnings
import numpy as np
from numpy.typing import NDArray
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader


class NemdDisplacementFrame:
    """Represents carbon-alpha atom displacements for a given frame.

    Attributes:
        carbon_alpha_index: indices specifying which carbon alpha atom each of
            the corresponding displacements are associated with. This is stored
            within the displacement file under the "CA" column. Note that this
            index iterates over the carbon alpha atoms in the base structure in
            the order in which they appear, and spans the domain [1, N_a] where
            "N_a" is the number of carbon-alpha atoms, not the total number of
            atoms present in the base structure.
        average_displacement: the mean displacement vector as was calculated by
            averaging over all displacement sample vectors.
        sample_size: number of displacement samples.
        standard_error_std1: standard error of the displacement vector using a
            standard deviation of one.
        standard_error_std2: standard error of the displacement vector using a
            standard deviation of two.
        standard_error_of_norm_std1: standard error of the norm of the displacement
            vector using a standard deviation of one.
        standard_error_of_norm_std2: standard error of the norm of the displacement
            vector using a standard deviation of two.


    """
    def __init__(
            self, carbon_alpha_index: NDArray[np.int64],
            average_displacement: NDArray[np.float64],
            sample_size: NDArray[np.int64],
            standard_error_std1: NDArray[np.float64],
            standard_error_std2: NDArray[np.float64],
            standard_error_of_norm_std1: NDArray[np.float64],
            standard_error_of_norm_std2: NDArray[np.float64]
    ):
        """Initialise the `NemdDisplacementFrame` instance.

        Arguments:
            carbon_alpha_index: indices specifying which carbon alpha atom each of
                the corresponding displacements are associated with. This is stored
                within the displacement file under the "CA" column. Note that this
                index iterates over the carbon alpha atoms in the base structure in
                the order in which they appear, and spans the domain [1, N_a] where
                "N_a" is the number of carbon-alpha atoms, not the total number of
                atoms present in the base structure.
            average_displacement: the mean displacement vector as was calculated by
                averaging over all displacement sample vectors.
            sample_size: number of displacement samples.
            standard_error_std1: standard error of the displacement vector using a
                standard deviation of one.
            standard_error_std2: standard error of the displacement vector using a
                standard deviation of two.
            standard_error_of_norm_std1: standard error of the norm of the displacement
                vector using a standard deviation of one.
            standard_error_of_norm_std2: standard error of the norm of the displacement
                vector using a standard deviation of two.
        """

        self.carbon_alpha_index = carbon_alpha_index
        self.average_displacement = average_displacement
        self.sample_size = sample_size

        self.standard_error_std1 = standard_error_std1
        self.standard_error_std2 = standard_error_std2

        self.standard_error_of_norm_std1 = standard_error_of_norm_std1
        self.standard_error_of_norm_std2 = standard_error_of_norm_std2

    @property
    def average_displacement_norm(self):
        """Norms of the average displacement vectors"""
        return np.linalg.norm(self.average_displacement, axis=1)

    @classmethod
    def load(cls, path: str) -> 'NemdDisplacementFrame':
        """Load displacement data from the specified file path.

        This will parse displacement data from a provided displacement file.
        Note that no specification currently exists for the "displacement"
        formatted files. Thus, a heuristic parser has been implemented until
        a codified specification has been ratified.

        Arguments:
            path: path to the displacement file that is to be loaded.

        Returns:
            frame: a `NemdDisplacementFrame` entity representing the data
                contained within the specified file.
        """

        # Set up lists to hold the file data
        carbon_alpha_index = []

        average_displacement = []

        sample_size = []

        standard_error_std1 = []
        standard_error_std2 = []

        standard_error_of_norm_std1 = []
        standard_error_of_norm_std2 = []

        # Open the specified file and loop over each line
        with open(path, "r") as file:
            for line in file:

                # Skip over blank lines and comment lines starting with "#".
                if line.startswith("#") or line.strip() == "":
                    continue

                # Sanitise the data to remove any "N/A" instances.
                line = line.replace("n.a.", "0.0")

                # Split the line into its individual numerical elements.
                ln = line.split()

                # Pull out and append the relevant data to the various lists.
                carbon_alpha_index.append(ln[0])

                average_displacement.extend(ln[1:4])

                sample_size.extend(ln[4:7])

                standard_error_std1.extend(ln[8:11])
                standard_error_std2.extend(ln[12:15])

                standard_error_of_norm_std1.append(ln[11])
                standard_error_of_norm_std2.append(ln[15])

        # Finally, perform any necessary type conversion and reshaping before
        # passing the data along to the `NemdDisplacementFrame` constructor.
        return cls(
            np.asarray(carbon_alpha_index, int),

            np.asarray(average_displacement, float).reshape((-1, 3)),

            np.asarray(sample_size, int).reshape((-1, 3)),

            np.asarray(standard_error_std1, float).reshape((-1, 3)),
            np.asarray(standard_error_std2, float).reshape((-1, 3)),

            np.asarray(standard_error_of_norm_std1, float),
            np.asarray(standard_error_of_norm_std2, float),
        )

    def generate_trajectory_frame(
            self, universe: MDAnalysis.Universe, confidence_level: int = 0
            ) -> NDArray[np.float32]:
        """Construct a trajectory frame from the displacement data.

        This method applies the displacements to a base structure, with a variable
        confidence interval, to generate an array of displaced positions.

        Arguments:
            universe: the MDAnalysis Universe object representing the base
                molecular structure upon which the displaced structures will
                be based.
            confidence_level: specifies the confidence level to use. If set to
                0, all displacements are included. Higher values apply stricter
                confidence criteria based on standard errors. A value of 1 uses
                a confidence interval of one standard deviation (68%), whereas
                a value of 2 will use a confidence interval of two standard
                deviations.

        Returns:
            NDArray[np.float32]: An array of atomic positions with displacements applied.

        Raises:
            UserWarning: If the system contains more than 20,000 atoms, a warning
                is issued as the NanoVer protocol is unable to process such systems.

        """

        if universe.atoms.n_atoms >= 20_000:
            warnings.warn("NanoVer protocol is unable to process systems with"
                          " more than 20,000 atoms.")

        # Construct an index array map that translates "carbon-alpha" indices,
        # as specified in the displacement file, to atomic indices as found in
        # the structure file.
        carbon_alpha_index_to_atomic_index = np.where(
            universe.atoms.names == "CA")[0]

        # Identify which displacements are statistically significant.
        threshold = {0: 0, 1: self.standard_error_of_norm_std1,
                     2: self.standard_error_of_norm_std2}[confidence_level]

        source_indices = np.where(self.average_displacement_norm > threshold)[0]

        # 3.1 Generate a copy of the atoms object.
        positions = universe.atoms.positions.copy()

        # 3.2 Loop over indices of statistically significant displacements.
        for source_index in source_indices:

            # Identify which atom this displacement is associated with
            atomic_index = carbon_alpha_index_to_atomic_index[
                self.carbon_alpha_index[source_index] - 1]

            # Add the displacement to the position data
            positions[atomic_index] += self.average_displacement[source_index]

        return positions

    @staticmethod
    def subsume_displacement_trajectories_into_universe(
            displacements: list["NemdDisplacementFrame"],
            universe: MDAnalysis.Universe,
            confidence_level: int = 0):

        """Integrates multiple displacement frames into the Universe trajectory.

         This method takes a list of `NemdDisplacementFrame` instances and uses
         them to create new trajectory frames for the specified Universe object.

         Args:
             displacements: list of `NemdDisplacementFrame` instances from which
                the frames that will make up the new trajectory will be generated.
             universe: the MDAnalysis Universe object representing the molecular
             structure.
             confidence_level: Specifies the confidence level to use for
                 generating the trajectory frames. If set to 0, all displacements
                 are included. Higher values apply stricter confidence criteria
                 based on standard errors. A value of 1 uses a confidence interval
                 of one standard deviation (68%), whereas a value of 2 will use
                 a confidence interval of two standard deviations.

         """

        frames = np.stack([
            i.generate_trajectory_frame(universe, confidence_level)
            for i in displacements])

        memory_reader = MemoryReader(frames, order="fac")

        universe.trajectory = memory_reader


    @staticmethod
    def auto_load_displacement_frames(path: str) -> list["NemdDisplacementFrame"]:
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