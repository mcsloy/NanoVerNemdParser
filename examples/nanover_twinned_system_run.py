import MDAnalysis
from dnemd.nanover.generators import DoubledGenerator
from dnemd.nanover.nemd_playback import TrajectoryPlayback
from dnemd.parsing.pdb import load_pdb_file_as_doubled_mdanalysis_topology

if __name__ == "__main__":
    """Example parser for sending NEMD displacement data to NanoVer-IMD.

    This script uses a custom file parser entity, `DisplacementFrames`, to load
    in and represent the D-NEMD displacement data. This than can be wrapped in
    a `SimpleGenerator` class to make it accessible to `MDAnalysis.Universe`
    entities.
    
    The `MDAnalysis` based trajectories may then be hosted by NanoVer using a
    traditional server structure, an interface for which is provided by the
    `TrajectoryPlayback` class.

    Usage:
        Specify the location of the base structural file using the variable
        `reference_structure_file_path`. The HDF5 file storing the D-NEMD data
        should then be assigned to the variable `displacement_file`.

        Other trajectory and visualisation options can be controlled by the
        supplementary variables provided in the code shown below. Note that
        documentation for these options is provided in the comments rather
        than within this docstring.  
    """

    # ╔════════════════════╗
    # ║      Settings      ║
    # ╚════════════════════╝
    # Note that may of the visualisation related settings, such as the choice of
    # colour map, may be changed in an ad-hoc manner while the server is running.
    #
    # Path to the reference structure file
    reference_structure_file_path = r"protein_file_path.pdb"
    # Paths to the hdf5 files storing the D-NEMD displacement data
    displacement_file_1 = r"path/to/the/displacement/data/file_1.h5"
    displacement_file_2 = r"path/to/the/displacement/data/file_2.h5"

    # In some situations it may be desirable to offset the position of the second
    # structure with respect to the first. If this is set to `None` then the two
    # structures will lie directly ontop of one another. Offsets can be defined
    # using a numpy a ray like so `offset = np.array([20., 0., 0.])`.
    offset = None

    # Given that the "two" systems will more or less lie ontop of one another
    # it can be somewhat hard to determine where one system starts and the other
    # ends. For this reason one may change the transparency of the proteins by
    # adjusting the alpha value. The alpha value should be assigned a value
    # within the domain [0, 1], with values closer to `1.0` being more opaque
    # and chose closer to `0.0` being more transparent. This will effectively
    # override the alpha values defined in the colour map. If you do not wish
    # this to happen then set this to `None` instead.
    alpha = 0.5

    # Trajectory frames shown per second.
    fps = 15

    # Should MDAnalysis attempt to infer atomic bonds?
    # The NanoVer client requires a bond list to utilise the "cartoon" renderer.
    # Some structure files include bond data, but not all do. If bond data is
    # absent from the structural file, set this flag to `True` to instruct
    # MDAnalysis to infer the bonds. Note that for larger systems, this process
    # can be extremely time-consuming. Therefore, it is recommended to perform
    # this inference once and then save the file with the bond data for future
    # use using the command `universe.atoms.write('protein_with_bonds.pdb', bonds='all')`.
    should_compute_bonds = False

    # Specifies what type of renderer is to be used to visualise the structure
    # by NanoVer clients. By default, a cartoon based renderer is used. Note
    # that changing the renderer to anything other than "cartoon extended"
    # will result in a loss of residue specific colouring & scaling features.
    renderer = "cartoon extended"

    # Name of the colour map to be used when colouring the protein residues
    # according to the displacement metric. This should be the name of a valid
    # MatPlotLib colour gradient. For a list of available colour maps see
    # matplotlib.org/stable/users/explain/colors/colormaps.html.
    colour_map_name = "viridis"

    # Given that some displacements can be too small to easily see, it is
    # sometimes best to scale them to magnify the visual effects. Atomic
    # displacements will be scaled by the factor specified below before being
    # added to the reference frame. Note that this scaling can be done
    # dynamically while the simulation is running via either the
    # `TrajectoryGenerator` or `TrajectoryPlayback` entities.
    displacement_scale_factor = 1.0

    # If the "nemd" renderer is used, each residue will be coloured according
    # to its displacement. The following variables set the initial range of the
    # colour map. Ideally, the minimum and maximum values should correspond to
    # the minimum and maximum displacements. It is important to note that the
    # displacement values used for sampling the colour map do not include the
    # `displacement_scale_factor`. That is to say if the maximum displacement
    # present is 0.8 then `colour_metric_normalisation_max` should be set to
    # `0.8` and not `0.8 * displacement_scale_factor`. If These values are not
    # assigned then an estimation will be made later on in this script.
    colour_metric_normalisation_min = None
    colour_metric_normalisation_max = None

    # The metric used for colouring the residues may also be utilised to adjust their
    # scale, making displacements more noticeable. The size of each residue is determined
    # by standard linear interpolation using the formula `l + (u - l) * x`, where `l`
    # and `u` are the lower and upper bounds as defined by `residue_scale_from` and
    # `residue_scale_to`, respectively. The value `x` is the same normalised metric
    # produced during colour mapping. The final formula is:
    #   `residue_scale_from + (residue_scale_to - residue_scale_from) *
    #   (displacement - colour_metric_normalisation_min) /
    #   (colour_metric_normalisation_max - colour_metric_normalisation_min)`
    #
    # It is recommended to keep `residue_scale_from` at `1.0`. To make scaling re
    # pronounced, increase the `residue_scale_to` setting. Per-residue scaling can
    # be disabled by setting `residue_scale_to` to `1.0`.
    residue_scale_from = 1.0
    residue_scale_to = 4.0

    # The NanoVer session will not be recorded unless a file name is provided to
    # the `TrajectoryPlayback` instance. By default, no recording is made. If a
    # file path is specified, the session will be recorded and saved in a pair
    # of files, namely "<FILE_NAME>.traj" and "<FILE_NAME>.state".
    record_to_file = None

    # ╔════════════════════╗
    # ║   Initialisation   ║
    # ╚════════════════════╝
    #
    # Construct a `Universe` entity and load in the structure file. Note that
    # if the file does not contain information about bonds, then MDAnalysis
    # must be instructed to calculate the bonds by setting the `guess_bonds`
    # flag.
    universe = MDAnalysis.Universe(
        load_pdb_file_as_doubled_mdanalysis_topology(
            reference_structure_file_path),
        guess_bonds=should_compute_bonds)

    # Construct the trajectory entity, and assign it to the universe object
    trajectory = DoubledGenerator([displacement_file_1, displacement_file_2], offset=offset)
    universe.trajectory = trajectory

    # If not scaling bounds were supplied then just set them to the maximum and
    # minimum displacement values.
    if not colour_metric_normalisation_min:
        colour_metric_normalisation_min = trajectory.minimum_displacement_distance
    if not colour_metric_normalisation_max:
        colour_metric_normalisation_max = trajectory.maximum_displacement_distance

    # ╔════════════════════╗
    # ║      Playback      ║
    # ╚════════════════════╝
    #
    # Construct the trajectory playback entity
    trajectory_player = TrajectoryPlayback(
        universe, fps=fps,
        colour_metric_normalisation_min=colour_metric_normalisation_min,
        colour_metric_normalisation_max=colour_metric_normalisation_max,
        residue_scale_from=residue_scale_from,
        residue_scale_to=residue_scale_to,
        colour_map_name=colour_map_name,
        alpha=alpha,
        record_to_file=record_to_file)

    # Publish the topology data
    trajectory_player.send_topology_frame()
    # Initiate playback
    trajectory_player.play()

    # ╔════════════════════╗
    # ║      Visuals       ║
    # ╚════════════════════╝
    #
    # Enable cartoon rendering mode, or whatever was specified in the settings section.
    trajectory_player.set_global_renderer(renderer)

    # Scale the displacements by the following factor to make them more evident
    trajectory_player.displacement_scale_factor = displacement_scale_factor

    # ╔════════════════════╗
    # ║         Run        ║
    # ╚════════════════════╝
    #
    # Do not terminate the python thread, and therefore the server, until the user
    # expressly indicates that it safe to do so via depression of the return key.
    prompt = input("> Press the return key to terminate the server...")
    trajectory_player.pause()
    trajectory_player.frame_server.close()
    exit()
