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
The `nanover_single_system_run.py` file, located in the examples directory of this repository, contains the code required to visualise D-NEMD data using NanoVer in a self-describing format. It is the recommended starting point for users looking to visualise D-NEMD data with NanoVer. The rest of this section is devoted to explaining the various components of the `dnemd` package that form the foundational boilerplate code enabling these examples.

## Minimum Working Example
The `TrajectoryPlayback` class is responsible for managing the playback of D-NEMD simulation data using the NanoVer visualiser. To instantiate the `TrajectoryPlayback` class, only one mandatory argument, `universe`, is required. The `universe` is an `MDAnalysis.Universe` object that encapsulates the structure and trajectory of the system of interest. 

Trajectory generators in `MDAnalysis.Universe` instances are crucial for reading molecular dynamics trajectories from diverse file formats. These generators are generally based on the `ReaderBase` class, providing a standard interface for accessing trajectory data. Different trajectory generators are employed by MDAnalysis to accommodate the specific requirements of each file format. Consequently, the introduction of a new file format necessitates the creation of a custom trajectory generator (`SimpleDNemdGenerator`) to ensure proper data handling. Thus, the minimum viable code necessary to playback D-NEMD data using NanoVer is as follows:
```python
from MDAnalysis import Universe
from dnemd.nanover.generators import SimpleDNemdGenerator
from dnemd.nanover.nemd_playback import TrajectoryPlayback
# Instantiate a `Universe` instance representing the base structure, using a reference file
universe = Universe("path/to/reference/structure/file.pdb")
# Create a generator that can produce trajectory frames from the D-NEMD data
trajectory = SimpleDNemdGenerator("path/to/dnemd/displacement/file.h5")
universe.trajectory = trajectory
# Finally, set up the trajectory player and start playback
trajectory_player = TrajectoryPlayback(universe)
trajectory_player.play()
```
It is important to note that for the time being the `SimpleDNemdGenerator` requires displacement data to be stored at the root node of the HDF5 file. Invoking the `play` method will start playback, which can then be stopped via the `pause` method. General playback operations are pushed to a background thread. Thus care must be taken to terminate the background threads when exiting the script; this can be done via `trajectory_player.frame_server.close()`.

## Visualisation Settings
The `TrajectoryPlayback` class offers numerous configurable settings to control the visual representation of D-NEMD data within NanoVer. However, most of these settings apply specifically to the `"cartoon extended"` renderer. Therefore, users should ensure that the correct renderer is set by calling `trajectory_player.set_global_renderer("cartoon extended")` before proceeding. Visual settings can be adjusted either via keyword arguments during instantiation or dynamically at runtime.

The `fps` (frames per second) setting controls the playback speed of the trajectory. It determines how many trajectory frames are displayed per second during the visualisation. A higher `fps` results in smoother, faster playback, while a lower `fps` slows down the visualisation, making it easier to observe individual frames. Adjusting this setting allows users to fine-tune the playback speed to match their needs for real-time observation or detailed analysis. By default, the trajectory will loop once it reaches the last frame.

The `displacement_scale_factor` setting controls the overall magnitude of the visualised displacements, allowing users to amplify or reduce the apparent motion of residues. By default, displacements are shown at their actual scale, but by increasing the `displacement_scale_factor`, the displacements can be magnified to make subtle movements more noticeable. Conversely, lowering the factor will minimise the visual effect of the displacements. This setting is applied globally, affecting all residues uniformly. It is particularly useful for highlighting small displacements that might otherwise be difficult to see, or for adjusting the visual impact in crowded or complex systems. The `displacement_scale_factor` does not affect the normalisation used for colouring and scaling (as discussed below), allowing it to be adjusted independently for clearer visualisation.

### Per-residue Settings
To visually highlight areas of high activity, each residue within the protein can be coloured and scaled based on the relative magnitude of its displacement. The displacement values are normalised and mapped to a chosen colour gradient, allowing residues with larger displacements to stand out. Additionally, residues can be scaled in size, providing further visual emphasis. This scaling, combined with the colour mapping, helps make displacement patterns more evident in the molecular structure.

#### Colour
To control the colouring of residues, a Matplotlib colour map must be specified using the `colour_map_name` parameter. This defines the gradient used to map displacement magnitudes to colours, with common options like 'viridis' or 'plasma'. The displacement values are normalised between `colour_metric_normalisation_min` and `colour_metric_normalisation_max`, which set the range for the mapping. Residues with displacement values at or below the minimum are coloured using the start of the gradient, while those with displacements at or above the maximum take the final colour in the gradient. An example showing how per-residue colouring can be configured is provided below:

```python
trajectory_player.set_global_renderer("cartoon extended")
trajectory_player.displacement_normalisation_lower_bound = trajectory.minimum_displacement_distance
trajectory_player.displacement_normalisation_upper_bound = trajectory.maximum_displacement_distance
trajectory_player.colour_map_name = "viridis"
```
Additional settings, such as `colour_metric_normalisation_power`, can be used to apply a non-linear transformation to the normalisation process, making differences in lower or higher displacement ranges more or less pronounced. Together, these parameters offer fine control over how displacement data is visually represented through residue colour. The `TrajectoryPlayback` class's `minimum_displacement_distance` and `maximum_displacement_distance` properties can be useful for helping to set up the correct normalisation bounds. While these settings may be configured at runtime, they may also be provided as arguments to the `TrajectoryPlayback` class constructor.

The `"cartoon extended"` renderer will respect alpha transparency values produced by the supplied colourmap. Users wishing to override this behaviour may do so using the `alpha` setting. 

#### Scale
Residue scaling allows the size of each residue to be adjusted based on its displacement, making regions of higher activity visually larger. The scaling is controlled by two parameters: `residue_scale_from` and `residue_scale_to`. These define the range within which the size of the residues will vary. A residue with minimal displacement will be scaled to the value specified by `residue_scale_from` (usually set to 1.0 for no scaling), while residues with the highest displacements will be scaled up to `residue_scale_to`.

The displacement values are normalised based on the same range used for colouring, ensuring consistency between visual effects. This scaling can emphasise areas of significant motion, drawing attention to regions of interest within the molecular structure. By adjusting the scaling range, users can increase or decrease the visual prominence of displacements. Only two settings need to be set to enable this feature:

```python
trajectory_player.residue_scale_minimum = 1.0
trajectory_player.residue_scale_maximum = 4.0
```

## Twinned Playback
It is possible to overlay the playback of two different D-NEMD displacements simultaneously by using the `DoubledGenerator` trajectory generator instead of the `SimpleDNemdGenerator`, as demonstrated in the twinned-system example script. Please note that this is a highly experimental feature and may undergo significant changes. To enable this functionality, the reference system must be doubled, meaning all atoms in the system are duplicated. This can be achieved with the `load_pdb_file_as_doubled_mdanalysis_topology` method, as illustrated in the twinned-system example script. The obvious caveat here being that the displacements must be with respect to the same underlying reference system. 
