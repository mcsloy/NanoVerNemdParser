# Introduction
This repository presents a provisional HDF5-based schema for storing and managing results derived from the analysis of Dynamical Non-Equilibrium Molecular Dynamics (D-NEMD) simulations. The schema's specifications are detailed in the *Hierarchical File Format Schema for D-NEMD Results* section of this document, while a Python implementation is available in the accompanying `dnemd` package. The `dnemd` Python package also contains all of the the code necessary to renderer the D-NEMD data using the NanoVer virtual reality visualiser application. The first section of this document is dedicated to the file schema, with the following section detailing the NanoVer related components.

# Hierarchical File Format Schema for D-NEMD Results
This section outlines a provisional HDF5 based schema for the storage and manipulation of results stemming from the analysis of Dynamical NonEquilibrium Molecular Dynamics (D-NEMD) simulations. 

## Schema
HDF5 is a hierarchical file format designed to store and organise large amounts of data. It allows for the efficient management of complex datasets, enabling the storage of multiple types of data within a single file. The format supports metadata, enabling users to annotate data and facilitate understanding and reuse. HDF5 is widely used for scientific data due to its flexibility, portability, and ability to handle large-scale datasets efficiently. Data are organised into groups and datasets, allowing for logical structuring and easy access.

As of schema version `1.0`, each entry consists of a single `group`, eleven `datasets`, and four metadata `attributes`. All of which are detailed in the following subsections.

### Datasets
All of the `datasets` are stored within the entry's root `group`. The `datasets`, listed in the following table, can be roughly partitioned into one of three data categories, as outlined below.

| Name                                 | Type  | Shape               | Description                                             |
| ------------------------------------ | ----- | ------------------- | ------------------------------------------------------- |
| `reference_structure_atomic_numbers` | int   | $n$                 | Atomic numbers of the reference structure atoms         |
| `reference_structure_positions`      | float | $n\times 3$         | Positions of atoms in the reference structure           |
| `atomic_indices`                     | int   | $m$                 | Indices of atoms associated with displacements          |
| `displacement_vectors`               | float | $k\times m\times 3$ | Mean displacement vectors for tracked atoms over frames |
| `displacement_norms`                 | float | $k$                 | Euclidean norms of the displacement vectors             |
| `sample_sizes`                       | int   | $k\times m\times 3$ | Number of samples taken for each atom in x, y, and z    |
| `standard_error_1_vectors`           | float | $k\times m\times 3$ | Standard error of displacement vectors (1σ)             |
| `standard_error_1_norms`             | float | $k\times m$         | Standard error of displacement vector norms (1σ)        |
| `standard_error_2_vectors`           | float | $k\times m\times 3$ | Standard error of displacement vectors (2σ)             |
| `standard_error_2_norms`             | float | $k\times m$         | Standard error of displacement vector norms (2σ)        |
| `frame_times`                        | float | $k$                 | Simulation times for each frame                         |

#### Reference Structure
The **`reference_structure_atomic_numbers`** array, as the name suggests, is an integer array of length $n$, specifying the atomic number of each of the $n$ atoms in the reference structure. The **`reference_structure_positions`** array is a float array of size $n \times 3$, reporting the positions of these atoms. Together, these two arrays collectively define the reference structure linked to the nonequilibrium atomic displacement data reported in this entry. Typically, displacement data files are paired with a reference structure file, which provides detailed information about the equilibrium structure. However, the atomic numbers and positions of the atoms in the reference structure are stored within the displacement data. This allows the displaced positions to be calculated independently, without needing the reference structure file.
#### Displacements
The **`atomic_indices`** array is an integer array of length $m$ that specifies which atoms in the reference structure each displacement corresponds to. Here, $m$ is the number of atoms for which displacements are being tracked. This is required as displacements are commonly only tracked for a select subset of atoms. The **`displacement_vectors`** array stores the mean atomic displacement vectors for each tracked atom across all frames. It is a thee dimensional float array of shape $k \times m \times 3$ , where $k$ represents the number of time frames, $m$ represents the number of tracked atoms, and the last dimension holds the displacement components (x, y, z) for each atom at each time step. The Euclidean norms of these displacement are stored within the float array **`displacement_norms`**. Finally, the number of samples used to calculate the mean displacement vectors, for each direction and atom at each time frame, is provided in the $k \times m \times 3$ integer array **`sample_sizes`**.
#### Errors
The $k\times m\times 3$ **`standard_error_1_vectors`** float array stores the standard errors of the displacement vectors, calculated using a sigma value of one. The corresponding standard errors of the displacement norms are stored in **`standard_error_1_norms`**. Similarly, **`standard_error_2_vectors`** and **`standard_error_2_norms`** store the standard errors calculated using a sigma value of two.
#### General
The **`frame_times`** array specifies the simulation time for each frame. Specifically, **`frame_times[k]`** gives the time at which the displacement vectors in **`displacement_vectors[k, :, :]`** were recorded for frame `k`.

### Attributes
The current schema allows for up to four general metadata`attributes`, all of which, if present, are attached to the root `group` of the entry. The first three `attributes`, listed in the table below, are considered optional and can be user-defined. The **`name`** `attribute` allows for the assignment of a human-readable name to the entry, while the **`spatial_units`** and **`temporal_units`** attributes specify the units for spatial data, such as displacements, and temporal data, such as frame times respectively. It is important to note that these `attributes` are entirely optional and serve no functional purpose beyond record keeping for the user’s convenience. Unlike those previously mentioned, the fourth and final `attribute`, **`version`**, is mandatory and reserved for use by the parser to record the schema version number.

| Name             | Type  | Description                                 |
| ---------------- | ----- | ------------------------------------------- |
| `name`           | `str` | Optional label for the displacement dataset |
| `spatial_units`  | `str` | Units of spatial data (e.g., displacements) |
| `temporal_units` | `str` | Units of temporal data (e.g., frame times)  |
| `version`        | `str` | Schema version number.                      |

## Python Interface
A Python based interface for this schema is provided by the class "`DisplacementFrames`", found within the `dnemd.parsing.schema` module. It is designed to allow users to efficiently read, write, or otherwise manipulate D-NEMD-related data in a manner consistent with the proscribed schema. The interface is structured similarly to a standard data-class, albeit one supplemented with basic serialisation functionality.

### Instantiation
A `DisplacementFrames` instance can be instantiated in one of three main ways, as outlined below. Firstly an instance can be instantiated directly by passing the eleven mandatory data arrays into the class's constructor method like so:
```python
from dnemd.parsing.schema import DisplacementFrames

frames = DisplacementFrames(
    reference_structure_atomic_numbers, reference_structure_positions,
    atomic_indices,
    displacement_vectors, displacement_norms,
    sample_sizes,
    standard_error_1_vectors, standard_error_1_norms,
    standard_error_2_vectors, standard_error_2_norms,
    frame_times)
```

It should be noted that the supplied arrays are expected to be instance of the numpy `NDArray` datatype. The optional `name`, `spatial_units`, and `temporal_units` metadata attributes may also be provided as keyword arguments as and when needed. The mandatory data arrays, as outlined in the previous section, may then be accessed via their associated attribute. The `DisplacementFrames` class also provides some ancillary properties, namely:
- `number_of_frames`: number of displacement frames stored ($k$).
- `number_of_displacements`: number of displacements reported per frame ($m$).
- `number_of_atoms_in_reference_structure`: specifies number of atoms present in the reference structure ($n$).
- `is_memory_mapped`: a Boolean indicating if the data is memory-mapped; more information will be provided on memory-mapping later on.

Alternatively, `DisplacementFrames` instances may be loaded from an appropriately structured HDF5 `Group` instance using the `read_from_hdf5_group` class method. An example demonstrating this is provided in the following code block:
```python
from dnemd.parsing.schema import DisplacementFrames
import h5py

path_to_hdf5_file = "/path/to/the/hdf5/file.h5"
group_path_and_name = "some_group_name"

with h5py.File(path_to_hdf5_file, "r") as file:
    group = file[group_path_and_name]
    frames = DisplacementFrames.read_from_hdf5_group(group)
```

When working with very large datasets, loading all the data into memory at once may be impractical or even impossible. For this reason, the `DisplacementFrames` class explicitly supports the HDF5 memory-mapping feature. This feature may be enabled by setting the `memory_mapped` flag to `True` when invoking the `read_from_hdf5_group` method. This will keep the data arrays as `h5py.Datasets`, which permits indexing to take place without loading the entire array into memory. However, it is worth noting that the associated HDF5 file must remain open for duration of the instance's lifetime. When used in memory-mapped mode, the associated `h5py.File` instance may be accessed via the `get_memory_mapped_file` method. 

If an HDF5 contains only a single D-NEMD entry, and that entry is stored at the root node, then the `read` may be used to instantiate a `DisplacementFrames` instance from a file path:
```python
from dnemd.parsing.schema import DisplacementFrames

path_to_hdf5_file = "/path/to/the/hdf5/file.h5"
# Only works when the entry's data is stored in the root node.
frames = DisplacementFrames.read(path_to_hdf5_file)
```

### Serialisation
Just as the `read_from_hdf5_group` method can be used to create a `DisplacementFrames` instance from an HDF5 `Group`, the `write_to_hdf5_group` method may be employed to serialise a `DisplacementFrames` object into a given HDF5 `Group`, as demonstrated below:
```python
from dnemd.parsing.schema import DisplacementFrames

path_to_hdf5_file = "/path/to/the/hdf5/file.h5"
group_path_and_name = "some_group_name"

frames = ...

with h5py.File(path_to_hdf5_file, "w") as file:
    group = file.create_group(group_path_and_name)
    frames = DisplacementFrames.write_to_hdf5_group(group)
```
Furthermore, the `write` method may be used to create a new HDF5 file and store the associated `DisplacementFrames` instance within it at the root node. Note that these serialisation methods are not compatible with memory-mapped `DisplacementFrames` instances. For users wishing to modify data while in memory mapped mode, they need only open the associated HDF5 file in `r+` mode.

### Miscellaneous
Files written in the now deprecated column-based structured file format following the "`average_xyz_displacement_<TIME>ps`" naming scheme may be converted to their HDF5 equivalent using the `dnemd.parsing.schema._convert_old_files` method. This is demonstrated below:
```python
from dnemd.parsing.schema import _convert_old_files
directory_path = r"path/to/reference/old_displacement_file_directory"
structure_file_path = r"path/to/reference/structure/file.pdb"
displacement_frames = _convert_old_files(directory_path, structure_file_path)
displacement_frames.write("new_file_name.h5")
```
Note that if file names do not match the original `"average_xyz_displacement_*ps"` naming convention, then a new regex pattern must be specified via the optional `pattern` argument.



# Visualisation
The `nanover_single_system_example.ipynb` file, located in the examples directory of this repository, contains the code required to visualise D-NEMD data using NanoVer in a self-describing format. It is the recommended starting point for users looking to visualise D-NEMD data with NanoVer. The rest of this section is devoted to explaining the various components of the `dnemd` package that form the foundational boilerplate code enabling these examples.
