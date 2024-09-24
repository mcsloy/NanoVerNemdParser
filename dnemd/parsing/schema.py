from typing import Optional, Self
import numpy as np
from packaging.version import Version
from h5py import File, Group, Dataset
from warnings import warn
from numpy.typing import NDArray


class DisplacementFrames:
    """Dynamical non-equilibrium molecular dynamics frame data container.

    This data class stores and writes displacement data resulting from Dynamical
    NonEquilibrium Molecular Dynamics (D-NEMD) simulations. Specifically, D-NEMD
    involves perturbing a system that has reached equilibrium to study its time-
    -dependent response to external influences. This class tracks the trajectory
    of atomic displacements in a subset of atoms within a reference system after
    such perturbations have been applied.

    The mean atomic displacement vectors for each of the tracked atoms at each
    time frame is stored within the attribute `displacement_vectors`. The atoms
    to which each set of displacement vectors is associated is specified by the
    `atomic_indices` attribute.

    The `DisplacementFrames` instances may be written to & read from HDF5
    formatted files via the `write` & `read` methods respectively. Alternatively,
    the `write_to_hdf5_group` and `read_from_hdf5_group` may be used by users
    wishing to store multiple such structures in a single HDF5 file. By default
    the contents of each file are read directly into memory. However, setting
    `memory_mapped=True` when opening a file will cause the data to be memory-
    -mapped for reduced memory overhead.

    It is important to note that when run in memory-mapped mode the attributes
    will not be numpy `NDArray` instances but rather h5py `Dataset` entities.

    Attributes:
        reference_structure_atomic_numbers: an array, of length R, specifying the
            atomic numbers of each atom present in the reference structure. Where
            R is the number of atoms in the reference structure.
        reference_structure_positions: the positions of each atom in the reference
            structure stored in a R×3 array.
        atomic_indices: indices specifying which atom each of the corresponding
            displacements are associated with. This is required as displacements
            may be only be provided for a subset of the atoms present in the
            reference structure.
        displacement_vectors: an array like entity storing the mean displacement
            vectors for each of the tracked atoms at each frame. This should be
            an N×T×3 array, where N represents the number of trajectory frames
            and T represents the number of tracked atoms.
        displacement_norms: an array, of length N, storing the Euclidean norm
            of the displacement vectors.
        sample_sizes: an N×T×3 array specifying the number of samples taken along
            each dimension (x,y,z) for each target atom at each timestep.
        standard_error_1_vectors: the standard error of the displacement vectors
            calculated using a standard deviation of one. This should be an
            array of shape N×T×3.
        standard_error_1_norms: the standard error of the norms of the displacement
            vectors calculated using a standard deviation of one. This should
            be an array of shape N×T.
        standard_error_2_vectors: the standard error of the displacement vectors
            calculated using a standard deviation of two. This should be an
            array of shape N×T×3.
        standard_error_2_norms: the standard error of the norms of the displacement
            vectors calculated using a standard deviation of two. This should
            be an array of shape N×T.
        frame_times: the simulation times corresponding to when each frame was
            recorded.
        name: an optional name, of type `str`, used to label the displacement
            dataset.
        spatial_units: an optional string specifying the units in which spatial
            data such as the displacements are reported.
        temporal_units: an optional string specifying the units in which
            temporal data such as the frame times are reported.

    Properties:
        number_of_frames: number of displacement frames stored.
        number_of_displacements: number of displacements reported per frame.
        number_of_atoms_in_reference_structure: specifies number of atoms
            present in the reference structure.
        is_memory_mapped: indicates if the data is memory-mapped.


    Notes:
        The parser considers data provided via the keyword arguments, such as
        units, to be descriptive metadata. As such, the units do not serve any
        functional purpose, other than record keeping.

        Users are free to add in any additional data they wish to the HDF5 file.
        However, such data will not automatically be parsed by this entity.

        The displacement data file represented by this class is typically
        accompanied by a reference structure file, which provides detailed
        information about the equilibrium structure. However, the atomic numbers
        & positions of the atoms within the reference structure are also stored
        here in the displacement data structure. This ensures that displaced
        positions can be calculated independently, without requiring access to
        the reference structure file.


    Todo:
        A consensus should be reached on how the standard error arrays should
        be treated. The possible treatments are as follows:
            1. Mandatory: The standard errors are required and must be present
               in every entry.
            2. Optional: The standard errors are considered important enough to
               be included within the parser but are optional and may not be
               specified.
            3. Ancillary: If present, the standard errors will be ignored by the
               parser, and users will be responsible for extracting them if
               needed. That is to say they will be treated like any other data
               stored within the HDF5 file.
            4. Need to add in safety checks to the init method.
    """

    version: Version = Version("1.0.0")

    def __init__(self,
                 reference_structure_atomic_numbers: NDArray[int],
                 reference_structure_positions: NDArray[float],
                 atomic_indices: NDArray[int],
                 displacement_vectors: NDArray[float],
                 displacement_norms: NDArray[float],
                 sample_sizes: NDArray[int],
                 standard_error_1_vectors: NDArray[float],
                 standard_error_1_norms: NDArray[float],
                 standard_error_2_vectors: NDArray[float],
                 standard_error_2_norms: NDArray[float],
                 frame_times: NDArray[float],
                 **kwargs):
        """
        Arguments:
            reference_structure_atomic_numbers: an array, of length R, specifying the
                atomic numbers of each atom present in the reference structure. Where
                R is the number of atoms in the reference structure.
            reference_structure_positions: the positions of each atom in the reference
                structure stored in a R×3 array.
            atomic_indices: indices specifying which atom each of the corresponding
                displacements are associated with. This is required as displacements
                may be only be provided for a subset of the atoms present in the
                reference structure.
            displacement_vectors: an array like entity storing the mean displacement
                vectors for each of the tracked atoms at each frame. This should be
                an N×T×3 array, where N represents the number of trajectory frames
                and T represents the number of tracked atoms.
            displacement_norms: an array, of length N, storing the Euclidean norm
                of the displacement vectors.
            sample_sizes: a T×3 array specifying the number of samples taken along
                each dimension (x,y,z) for each target atom.
            standard_error_1_vectors: the standard error of the displacement vectors
                calculated using a standard deviation of one. This should be an
                array of shape N×T×3.
            standard_error_1_norms: the standard error of the norms of the displacement
                vectors calculated using a standard deviation of one. This should
                be an array of shape N×T.
            standard_error_2_vectors: the standard error of the displacement vectors
                calculated using a standard deviation of two. This should be an
                array of shape N×T×3.
            standard_error_2_norms: the standard error of the norms of the displacement
                vectors calculated using a standard deviation of two. This should
                be an array of shape N×T.
            frame_times: the simulation times corresponding to when each frame was
                recorded.

        Keyword Arguments:
            name: an optional name, of type `str`, used to label the displacement
                dataset.
            spatial_units: an optional string specifying the units in which spatial
                data such as the displacements are reported.
            temporal_units: an optional string specifying the units in which
                temporal data such as the frame times are reported.


        """

        self.reference_structure_atomic_numbers = reference_structure_atomic_numbers
        self.reference_structure_positions = reference_structure_positions

        self.atomic_indices = atomic_indices

        self.displacement_vectors = displacement_vectors
        self.displacement_norms = displacement_norms

        self.sample_sizes = sample_sizes

        self.standard_error_1_vectors = standard_error_1_vectors
        self.standard_error_1_norms = standard_error_1_norms

        self.standard_error_2_vectors = standard_error_2_vectors
        self.standard_error_2_norms = standard_error_2_norms

        self.frame_times = frame_times

        self.name: Optional[str] = kwargs.get("name")
        self.spatial_units: Optional[str] = kwargs.get("spatial_units")
        self.temporal_units: Optional[str] = kwargs.get("temporal_units")

        # Ensure that the number of atomic numbers specified in the reference
        # structure matches up with the number of supplied positions.
        if (reference_structure_atomic_numbers.shape[0]
                != reference_structure_positions.shape[0]):
            raise IndexError(
                "Inconsistency detected in the number of atoms indicated by the"
                " length of the `reference_structure_atomic_numbers` and "
                "`reference_structure_positions` arrays.")

        # Varify that the arrays are of the correct length
        m = atomic_indices.shape[0]
        n = len(frame_times)
        check = self.__check_shape
        check(n, m, displacement_vectors, "displacement_vectors")
        check(n, m, standard_error_1_norms, "standard_error_1_norms")
        check(n, m, sample_sizes, "sample_sizes")
        check(n, m, standard_error_1_vectors, "standard_error_1_vectors")
        check(n, m, standard_error_1_norms, "standard_error_1_norms")
        check(n, m, standard_error_2_vectors, "standard_error_2_vectors")
        check(n, m, standard_error_2_norms, "standard_error_2_norms")

        # Ensure that the frame times array is ordered & contains no duplicates
        if not all(frame_times[:-1] < frame_times[1:]):
            raise ValueError(
                "The `frame_times` array must be ordered & contain no duplicates.")

    @staticmethod
    def __check_shape(n, m, array, name):
        expected_shape = (n, m, *array.shape[2:])
        if array.shape[0] != n or array.shape[1] != m:
            raise IndexError(
                f"Shape mismatch detected in array `{name}`; "
                f"expected {expected_shape}, but found {array.shape}.")

    # region Properties
    @property
    def number_of_frames(self) -> int:
        """Number of displacement frames"""
        return self.displacement_vectors.shape[0]

    @property
    def number_of_displacements(self) -> int:
        """Number of target atoms, & thus displacements, tracked in each frame."""
        return len(self.atomic_indices)

    @property
    def number_of_atoms_in_reference_structure(self) -> int:
        """Number of atoms present in the reference structure."""
        return len(self.reference_structure_atomic_numbers)

    @property
    def is_memory_mapped(self):
        """Boolean indicating whether the instance's data is memory mapped."""
        return isinstance(self.displacement_vectors, Dataset)

    # endregion

    # region IO Methods

    @classmethod
    def read(cls, file_path: str, memory_mapped=False, read_only=True) -> Self:
        """Read displacement data from an HDF5 file.

        This method creates an instance of the `DisplacementFrames` class by
        reading data from the specified HDF5 file. If the ``memory_mapped``
        option is enabled, the data will remain in memory-mapped mode, allowing
        access without loading the entire dataset into memory. For non-memory
        mapped cases, the data will be fully loaded into memory and the file
        will be automatically closed after reading.

        Arguments:
            file_path: path to the HDF5 file that is to be read.
            memory_mapped: If `True`, the data will be memory-mapped, enabling
                efficient access to large datasets. Default is `False`.
            read_only: by default, the file is read-only when opened in memory-
                -mapped mode. However, this restriction may be removed by
                setting the ``read_only`` flag to `False`. Note that this flag
                only applies to files that are loaded in memory-mapped mode.

        Returns:
            displacement_frames: a `DisplacementFrames` instance containing the
                parsed data.

        Raises:
            ValueError: when ``allow_writing`` is enabled but ``memory_mapped``
                is not.
        """
        # The enabling `allow_writing` makes no sense outside the context of a
        # memory-mapped file instance.
        if not memory_mapped and not read_only:
            raise ValueError(
                "The `allow_writing=True` is only valid for memory-mapped instances.")

        # If the file is to be parsed in memory-mapped mode then the h5py `File`
        # object should be left open.
        if memory_mapped:
            mode = "r" if read_only else "r+"
            return cls.read_from_hdf5_group(File(file_path, mode), memory_mapped)

        # If not, then ensure the file is closed following read operation via
        # a context.
        with File(file_path, "r") as file:
            return cls.read_from_hdf5_group(File(file, "r"), memory_mapped)

    @classmethod
    def read_from_hdf5_group(
            cls, source: File | Group, memory_mapped: bool = False) -> Self:
        """Instantiate a `DisplacementFrames` entity from an HDF5 entry.

        This constructor creates a new instance using data from the specified
        HDF5 `File` or `Group`. By default, the data is loaded into memory as
        NumPy arrays, enabling quick and easy access.

        For very large datasets, loading all data into memory at once may be
        impractical or impossible. In such cases, setting the ``memory_mapped``
        argument to `True` keeps the data as `h5py.Datasets`, allowing indexing
        without loading the entire array into memory. However, the associated
        HDF5 file must remain open for duration of the instance's lifetime.

        Arguments:
            - source: the HDF5 file or group from which the displacement data
                is to be sourced.

        Keyword Arguments:
            - memory_mapped: if `True`, data will remain as `h5py.Dataset`
                entities, suitable for large datasets that cannot fit into
                memory. Default is `False`.

        Returns:
            - displacement_frames: the corresponding `DisplacementFrames` entity.
        """

        def load_array(dataset_name):
            if memory_mapped:
                return source[dataset_name]
            else:
                return source[dataset_name][...]

        # Read the version information so that compatability can be verified.
        # Note that this is currently not in use, and will only be used when
        # and if a breaking change is made to the database format.
        version = Version(source.attrs["version"])

        # Issue a warning until something more involved is needed. Realistically,
        # this should never be called, as version handling will be implemented
        # as the same time as a new version.
        if version.release[:2] > cls.version.release[:2]:
            warn(
                "Schema version mismatch: The file was created with a newer "
                "schema version than\nthe currently installed library. "
                "Compatibility issues may arise.")

        # Load in the reference structure data
        reference_structure_atomic_numbers = load_array("reference_structure_atomic_numbers")
        reference_structure_positions = load_array("reference_structure_positions")

        # Load the displacement vectors, their norms, the atomic indices, and
        # the sample sizes.
        displacement_vectors = load_array("displacement_vectors")
        displacement_norms = load_array("displacement_norms")
        atomic_indices = load_array("atomic_indices")
        sample_sizes = load_array("sample_sizes")

        # Repeat for the standard error vectors and norms.
        standard_error_1_vectors = load_array("standard_error_1_vectors")
        standard_error_1_norms = load_array("standard_error_1_norms")
        standard_error_2_vectors = load_array("standard_error_2_vectors")
        standard_error_2_norms = load_array("standard_error_2_norms")

        # Finally load in the frame times
        frame_times = load_array("frame_times")

        # Get the system name, if provided, else default to `None`
        name = source.attrs.get("name")

        # Repeat for the units
        spatial_units = source.attrs.get("spatial_units")
        temporal_units = source.attrs.get("temporal_units")

        # Construct and return the `DisplacementFrames` instance.
        return cls(
            reference_structure_atomic_numbers,
            reference_structure_positions,
            atomic_indices, displacement_vectors, displacement_norms,
            sample_sizes, standard_error_1_vectors, standard_error_1_norms,
            standard_error_2_vectors, standard_error_2_norms, frame_times,
            name=name, spatial_units=spatial_units,
            temporal_units=temporal_units
        )

    def write(self, file_path: str):
        """Write displacement data to an HDF5 file.

        This method serialises the current `DisplacementFrames` instance into
        an HDF5 file. If the data is memory-mapped, writing is not allowed, and
        an error will be raised. To modify memory-mapped data, open the file
        with `allow_writing=True` and modify the data directly.

        Arguments:
            file_path: path to the file where the data will be written.

        Raises:
            RuntimeError: raised if the data is memory-mapped, as writing is
                not supported for memory-mapped instances.
        """

        # Ensure that the user is not making an attempt to write out a memory
        # mapped file.
        if self.is_memory_mapped:
            raise RuntimeError(
                "Cannot use the `write` method with memory-mapped files. Open "
                "the file with `read_only=False` to modify it directly; changes "
                "will be witten automatically."
            )

        # Create and open a new HDF5 file at the specified location.
        with File(file_path, "w") as file:
            # The place the data at the top level of the file.
            self.write_to_hdf5_group(file)

    def write_to_hdf5_group(self, target: File | Group):
        """Serialise a `DisplacementFrames` entity into an HDF5 entry.

        Arguments:
            - target: the HDF5 file or group into which the `DisplacementFrames`
                entity is to be serialised.

        """
        # The version used to write out the date is stored here. This makes it
        # easier to maintain backwards compatability should breaking changes be
        # made at a later date.
        target.attrs["version"] = str(self.version)

        # Save the reference structure data
        target.create_dataset("reference_structure_atomic_numbers", data=self.reference_structure_atomic_numbers)
        target.create_dataset("reference_structure_positions", data=self.reference_structure_positions)

        # Save the displacement vectors, their norms, and the atomic indices.
        target.create_dataset("atomic_indices", data=self.atomic_indices)
        target.create_dataset("displacement_vectors", data=self.displacement_vectors)
        target.create_dataset("displacement_norms", data=self.displacement_norms)
        target.create_dataset("sample_sizes", data=self.sample_sizes)

        # Repeat for the standard error vectors and norms.
        target.create_dataset("standard_error_1_vectors", data=self.standard_error_1_vectors)
        target.create_dataset("standard_error_1_norms", data=self.standard_error_1_norms)
        target.create_dataset("standard_error_2_vectors", data=self.standard_error_2_vectors)
        target.create_dataset("standard_error_2_norms", data=self.standard_error_2_norms)

        # Finally, store the frame times
        target.create_dataset("frame_times", data=self.frame_times)

        # Set the system name, but only if provided.
        if self.name:
            target.attrs["name"] = self.name

        # Repeat for the units
        if self.spatial_units:
            target.attrs["spatial_units"] = self.spatial_units

        if self.temporal_units:
            target.attrs["temporal_units"] = self.temporal_units

    # endregion

    # region General Methods

    def get_memory_mapped_file(self) -> File:
        """Retrieve the file associated with memory-mapped data.

        This method returns the open file from which the data is memory-mapped.
        If the instance is not memory-mapped, an error will be raised. This is
        only intended for use when the data is stored in `h5py.Dataset` entities.

        Returns:
            file: the hdf5 `File` entity to which this instance's data is mapped.

        Raises:
            FileNotFoundError: raised when the instance is not memory-mapped,
                meaning that no open file is associated with it.
        """

        if not self.is_memory_mapped:
            raise FileNotFoundError(
                "No open file is available as the instance is not memory-mapped.")
        else:
            # noinspection PyUnresolvedReferences
            return self.displacement_vectors.file

    # endregion

    # region Helper Methods

    def __repr__(self):
        """Return a string representation of the object."""
        # This function mostly exists so that something sensible is displayed
        # in the terminal.
        return f"{type(self).__name__}(" \
               f"number_of_frames={self.number_of_frames}, " \
               f"number_of_displacements={self.number_of_displacements}), " \
               f"memory_mapped={self.is_memory_mapped})"

    # endregion


def _convert_old_files(
        directory_path: str, structure_file_path: str,
        **kwargs) -> DisplacementFrames:
    """Convert old file format to new HDF5 structure.

    Arguments:
         directory_path: path to the directory in which the displacement data
            files are stored.
         structure_file_path: path to the reference structural file.

    Keyword Arguments:
        name: an optional name, of type `str`, used to label the displacement
            dataset.
        spatial_units: an optional string specifying the units in which spatial
            data such as the displacements are reported.
        temporal_units: an optional string specifying the units in which
            temporal data such as the frame times are reported.
        pattern: file naming pattern with a single wildcard character used to
            specify where in the file name the time-step is specified. This will
            default to `"average_xyz_displacement_*ps"`.

    Returns:
        updated_file_structure: a `DisplacementFrames` instance containing the
            parsed data. This can then be saved to a file using the relevant class
            methods.
    """
    from os.path import join, basename
    import re
    import glob
    from ase.io import read

    def locate_files():
        pattern = kwargs.get("pattern", "average_xyz_displacement_*ps")
        re_pattern = re.compile(fr"(?<={pattern.split('*')[0]})\d+(.\d*)?(?={pattern.split('*')[1]})")
        located_files = glob.glob(join(directory_path, pattern))
        located_files.sort(key=lambda i: int(re_pattern.search(basename(i)).group(0)))
        frame_times_list = [int(re_pattern.search(basename(i)).group(0)) for i in located_files]

        return located_files, frame_times_list

    # Define the lists into which the files' contents will be placed. This will
    # get turned into numpy arrays later on.
    carbon_alpha_index = []
    average_displacement = []
    sample_size = []
    std1, std2 = [], []
    std1_norm, std2_norm = [], []
    frame_times = []

    # Loop over each of the displacement frame files, in the correct order, and
    # parse them. The parsed data will then be appended to the previously
    # created lists.
    for file_name, frame_time in zip(*locate_files()):

        # Add frame time
        frame_times.append(frame_time)

        # Set up temporary lists to hold the file data
        carbon_alpha_index_t = []
        average_displacement_t = []
        sample_size_t = []
        std1_t, std2_t = [], []
        std1_norm_t, std2_norm_t = [], []

        # Open the specified file and loop over each line
        with open(file_name, "r") as file:
            for line in file:

                # Skip over blank lines and comment lines starting with "#".
                if line.startswith("#") or line.strip() == "":
                    continue

                # Sanitise the data to remove any "N/A" instances.
                line = line.replace("n.a.", "0.0")

                # Split the line into its individual numerical elements.
                ln = line.split()

                # Pull out and append the relevant data to the various lists.
                carbon_alpha_index_t.append(ln[0])
                average_displacement_t.extend(ln[1:4])
                sample_size_t.extend(ln[4:7])
                std1_t.extend(ln[8:11])
                std2_t.extend(ln[12:15])
                std1_norm_t.append(ln[11])
                std2_norm_t.append(ln[15])

        # Perform any necessary type conversion and reshaping before appending
        # them to their associated list.
        carbon_alpha_index.append(np.asarray(carbon_alpha_index_t, int) - 1)
        average_displacement.append(np.asarray(average_displacement_t, float).reshape((-1, 3)))
        sample_size.append(np.asarray(sample_size_t, int).reshape((-1, 3)), )
        std1.append(np.asarray(std1_t, float).reshape((-1, 3)))
        std2.append(np.asarray(std2_t, float).reshape((-1, 3)))
        std1_norm.append(np.asarray(std1_norm_t, float))
        std2_norm.append(np.asarray(std2_norm_t, float))

    # The old files do not supply the indices of the atoms associated with each
    # displacement. Rather they provide the carbon-alpha index number. This is
    # converted into an atomic index for ease of use.

    atoms = read(structure_file_path)
    atom_types = atoms.arrays["atomtypes"]
    alpha_to_atom_index_map = np.where(np.equal(atom_types, "CA"))[0]
    atomic_indices = alpha_to_atom_index_map[carbon_alpha_index[0]]

    return DisplacementFrames(
        atoms.get_atomic_numbers(),
        atoms.positions,
        atomic_indices, np.array(average_displacement),
        np.linalg.norm(average_displacement, axis=-1),
        np.array(sample_size),
        np.array(std1), np.array(std1_norm),
        np.array(std2), np.array(std2_norm),
        np.array(frame_times, dtype=float),
        **kwargs
    )


