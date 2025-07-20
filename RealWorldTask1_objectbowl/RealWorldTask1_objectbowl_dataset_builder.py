from typing import Iterator, Tuple, Any

import os
import PIL
import glob
import h5py
import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class RealWorldTask1_objectbowl(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.local_traj_data_path = '/projects/vidlab_data/data/xArm7_RealWorldTrajectories/task1_object-in-bowl' # base path to trajectories
        self.glob_str = '/projects/vidlab_data/data/xArm7_RealWorldTrajectories/task1_object-in-bowl/**/*.hdf5' # glob string that will return all of the .hdf5 files. Test this with `glob.glob(glob_str, recursive=True)`

        self.desired_hz = 5 # base sampling rate is 15hz
        
        self.clamp_num_trajectories = 30 # for controlling # training trajectories. set to None to use all trajectories

        # currently the .hdf5 files dont have the task name, so we parse it from the filename. Modify this if needed
        # might also need to update `'language_instruction': self.lang_dict[episode_path.split('/')[5]]` in the `_parse_example` function below
        self.lang_dict = {
            "place_the_banana_in_the_bowl": "Pick up the banana and place it in the white bowl",
            "place_the_eggplant_in_the_bowl": "Pick up the eggplant and place it in the white bowl",
            "place_the_green_pepper_in_the_bowl": "Pick up the green pepper and place it in the white bowl",
            "place_the_tomato_in_the_bowl": "Pick up the tomato and place it in the white bowl",
            "place_the_carrot_in_the_bowl": "Pick up the carrot and place it in the white bowl",
            "place_the_grape_in_the_bowl": "Pick up the grape and place it in the white bowl",
            "place_the_orange_in_the_bowl": "Pick up the orange and place it in the white bowl",
            "place_the_corn_in_the_bowl": "Pick up the corn and place it in the white bowl",
            "place_the_green_apple_in_the_bowl": "Pick up the green apple and place it in the white bowl",
            "place_the_strawberry_in_the_bowl": "Pick up the strawberry and place it in the white bowl"
        }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x gripper position, 3x gripper rotation].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x EEF position/orientation, '
                            '1x gripper status].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=self.glob_str)
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        def _parse_example(episode_path):
            with h5py.File(episode_path, 'r') as f:
                image_keys = ['image_path_exo1', 'image_path_wristcam']
                image_paths = {}

                for imkey in image_keys:
                    image_paths[imkey] = [
                        path.decode('utf-8').replace('/workspace/xarm-dataset', self.local_traj_data_path)
                        for path in f[imkey][:]
                    ]

                for imkey in image_keys:
                    for p in image_paths[imkey]:
                        assert os.path.exists(p)

                ## training images
                images = {k: [] for k in image_keys}

                for imkey in image_keys:
                    for img_path in image_paths[imkey]:
                        img = PIL.Image.open(img_path)
                        img = img.resize((256, 256), resample=0)
                        img_array = np.array(img)
                        if len(img_array.shape) == 2:
                            img_array = np.stack([img_array] * 3, axis=-1)
                        images[imkey].append(img_array)

                    image_paths[imkey] = np.array(image_paths[imkey])

                ## training action
                actions = np.concatenate([
                    f['abs_pos'][:], 
                    f['abs_rot'][:], 
                    f['gripper'][:][:, np.newaxis],
                ], axis=1)[:len(images[image_keys[0]])].astype(np.float32) # TODO: Sometimes there is 1 less image than actions. Find why this is

                # discritize gripper. Default trajectories set 0 to active 1 to inactive. We want 1=active -1=inactive
                actions[:, -1] = actions[:, -1] * -2 + 1 # 

                # Data is collected at 15hz. Sample it here
                desired_hz = self.desired_hz
                assert 15 % desired_hz == 0, 'desired sampling hz is not compatible with collected data frequency'
                sample_factor = 15 // desired_hz
                actions = actions[::sample_factor]
                for k in image_keys:
                    images[k] = images[k][::sample_factor]

                delta_actions = actions.copy()
                tmp = actions[1:, :-1] - actions[:-1, :-1] # T-1, 6
                delta_actions[:-1, :-1] = tmp # still T, 7
                delta_actions = delta_actions[:-1] # clip last sample. T-1, 7
                delta_actions[:, -1] = actions[1:, -1] # 

                # DEBUG: number of frames to shift gripper actions
                debug_additional_gripper_shift = -3
                delta_actions[:, -1] = np.roll(delta_actions[:, -1], debug_additional_gripper_shift)

            episode = []
            episode_len = len(images[image_keys[0]])
            for i in range(episode_len-1):
                exo_image = images['image_path_exo1'][i]
                wrist_image = images['image_path_wristcam'][i]
                delta_action = delta_actions[i]

                # remove zero actions. These are actions where 1) there is no cartesian or rotational change and 2) there is no gripper change
                if np.all(delta_action[:-1] == 0) and actions[i, -1] == actions[i-1, -1]:
                    continue

                proprio_state = np.concatenate([actions[i,:-1], [0], actions[i,-1:]]).astype(np.float32) # EEF_EULER. 0 is padding (see OpenVLA config.py)

                episode.append({
                    'observation': {
                        'image': exo_image,
                        'wrist_image': wrist_image,
                        'state': proprio_state,
                    },
                    'action': delta_action,
                    'discount': 1.0,
                    'reward': float(i == episode_len),
                    'is_first': i == 1,
                    'is_last': i == episode_len,
                    'language_instruction': self.lang_dict[episode_path.split('/')[-4]]
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            
            return episode_path, sample

        episode_paths = glob.glob(path, recursive=True)
        if self.clamp_num_trajectories is not None:
            episode_paths = self.sample_trajectories(episode_paths)

        for sample in episode_paths:
            yield _parse_example(sample)

    '''
    Only necessary if you use `self.clamp_num_trajectories`
    '''
    def sample_trajectories(self, episode_paths):
        assert self.clamp_num_trajectories < len(episode_paths)
        assert self.clamp_num_trajectories % 10 == 0, "10 sub-tasks here. # samples trajectories must be divisible by 10"
        
        random.shuffle(episode_paths)
        tmp = {k: [] for k in self.lang_dict.keys()}
        samples_per_subtask = self.clamp_num_trajectories // 10

        for path in episode_paths:
            for k in tmp.keys():
                if k in path:
                    tmp[k].append(path)
                    break
        
        new_paths = []
        for _, v in tmp.items():
            new_paths += v[:samples_per_subtask] # already random shuffled above, so just select the first elements

        return new_paths