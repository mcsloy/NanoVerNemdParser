# Using the NEMD Displacement Data Script

There should be four python files present within this archive. The first is the `run.py` script, this is the primary script that you will directly interact with. This provides a minimum implementation of the NEMD file parser, trajectory generator, and NanoVer server. The other files `parser.py`, `trajectory_generator.py`, and `nemd_playback.py` provide all of the necessary backend code to parse, interpret, and serve the NEMD trajectory data to the NanoVer client. To use this just set the required file paths within the `run.py` script and execute it.

## Overview

The script utilises a custom file parser entity, `NemdDisplacementFrame`, to load and represent NEMD displacement data. It can convert a sequence of such files into a trajectory suitable for representation by `MDAnalysis`. These trajectories can then be hosted by NanoVer using a traditional server structure, facilitated by the `TrajectoryPlayback` class.

## Usage Instructions

### Prerequisites

- Install MDAnalysis, NanoVer-IMD, and the NanoVer server.
- Ensure you have the required displacement and structure files.

### Configuration

1. **Reference Structure File**: Specify the path to the reference structure file in `reference_structure_file_path`.
   ```python
   reference_structure_file_path = r"protein_file_path.pdb"
   ```

2. **Displacement File Directory**: Provide the directory path where the NEMD displacement files are stored in `displacement_file_directory_path`.
   ```python
   displacement_file_directory_path = r"path/to/displacement/file/directory"
   ```

3. **Bond Inference**: Determine whether MDAnalysis should infer atomic bonds. If the structure file lacks bond data, set `should_compute_bonds` to `True`. Be aware that this process can be time-consuming for larger systems.
   ```python
   should_compute_bonds = False
   ```

4. **Displacement Scale Factor**: Set the scale factor for atomic displacements to magnify visual effects.
   ```python
   displacement_scale_factor = 4.0
   ```

5. **Renderer Type**: Specify the type of renderer to be used for visualising the structure by NanoVer clients.
   ```python
   renderer = "cartoon"
   ```

### Initialisation

1. **Load the Structure File**:
   ```python
   universe = MDAnalysis.Universe(reference_structure_file_path, guess_bonds=False)
   ```

2. **Load Displacement Data**:
   ```python
   displacement_frames = NemdDisplacementFrame.auto_load_displacement_frames(displacement_file_directory_path)
   ```
   Alternatively, displacement data can be loaded manually:
   ```python
   displacement_frames = list(map(NemdDisplacementFrame.load, [path_one, path_two, ...]))
   ```

3. **Create Trajectory**:
   ```python
   trajectory = TrajectoryGenerator.from_nemd_displacement_frames(displacement_frames, universe)
   universe.trajectory = trajectory
   ```

### Playback

1. **Initialise Trajectory Playback**:
   ```python
   trajectory_player = TrajectoryPlayback(universe, fps=30)
   ```

2. **Publish Topology Data**:
   ```python
   trajectory_player.send_topology_frame()
   ```

3. **Start Playback**:
   ```python
   trajectory_player.play()
   ```

### Visualisation

1. **Enable Renderer Mode**:
   ```python
   client = NanoverImdClient.autoconnect()
   client.subscribe_multiplayer()
   client.subscribe_to_frames()
   root_selection = client.root_selection
   root_selection.renderer = renderer
   root_selection.flush_changes()
   ```

2. **Scale Displacements**:
   ```python
   trajectory_player.displacement_scale_factor = displacement_scale_factor
   ```

### Running the Server

Prevent the Python thread, and thus the server, from terminating until the user indicates it is safe to do so:
```python
prompt = input("> Press the return key to terminate the server...")
trajectory_player.pause()
trajectory_player.frame_server.close()
exit()
```

## Warning

This script loads all frame data into memory using an `MDAnalysis.MemoryReader` instance. This may lead to significant memory usage. If this becomes problematic, consider using `numpy.memmap` arrays to reduce the memory footprint.