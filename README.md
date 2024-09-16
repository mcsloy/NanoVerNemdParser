# Using the NEMD Displacement Data Script

There should be four python files present within this archive. The first is the `run.py` script, this is the primary script that you will directly interact with. This provides a minimum implementation of the NEMD file parser, trajectory generator, and NanoVer server. The other files `parser.py`, `trajectory_generator.py`, and `nemd_playback.py` provide all of the necessary backend code to parse, interpret, and serve the NEMD trajectory data to the NanoVer client. To use this just set the required file paths within the `run.py` script and execute it.
