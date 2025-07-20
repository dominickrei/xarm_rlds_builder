## Installation
`pip install tensorflow tensorflow_datasets tensorflow_hub matplotlib plotly wandb`

### Test installation: build a dummy dataset

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Creating rlds builder for a new real-world task

**Rename Dataset**: Change the name of the folder from `RealWorldTask1_objectbowl` to the name of your task.
also change the name of `RealWorldTask1_objectbowl_dataset_builder.py` by replacing `RealWorldTask1_objectbowl` with your task's name (e.g. RealWorldTask2_stackcubes_dataset_builder.py)
and change the class name `RealWorldTask1_objectbowl` in the same file to match your task's name.

**Update Paths**: In the `__init__` method update the base path to the trajectories (`self.local_traj_data_path`) that contains all images & .hdf5 actions, and the glob string (`self.glob_str`) that will return all .hdf5 actions.

**(Optional) Change sampling hz**: The trajectories are recorded at 15hz. If you want to downsample them, change `self.desired_hz`.

**(Optional) Update sampling function**: If your desired train samples is < total trajectories set `self.clamp_num_trajectories` to your desired train samples. Youll need to update `sample_trajectories()`.

**Build the dataset**: Run `tfds build --overwrite` and the dataset will be saved to `~/tensorflow_datasets/`


This is modified from kpertsch's [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) guide. See it for more detailed information (e.g., transforms, parallel processing).