## RBSRICNN testing:
- ### Datasets:
	- The datasets, used to train/evaluate the models, can be downloaded [here](https://github.com/goutamgmb/deep-burst-sr#datasets). 
- ### Testing:
	#### For Track-1:
	- Run file named "bsricnn_synsr_val.py" to reproduce validalition SR results for the Burst SR track-1. 
	- Run file named "bsricnn_synsr_test.py" to reproduce test SR results for the Burst SR track-1.
	#### For Track-2:
	- Run file named "bsricnn_realsr_val.py" to reproduce validalition SR results for the Burst SR track-2.
	- Run file named "bsricnn_realsr_test.py" to reproduce test SR results for the Burst SR track-2.

- ### Contained Directories information:
	- models: BSRICNN Network structures.
	- trained_nets_x4: BSRICNN trained network.
	#### For Track-1:
	- track1_val_set: Given LR bursts of validalition-set of track-1.
	- sr_results_track1_val_set: saved output SR images of our network for track1_val_set.
	- track1_test_set: Given LR bursts of testset of track-1.
	- sr_results_track1_test_set: saved output SR images of our network for track1_test_set.
	#### For Track-2:
	- track2_val_set: Given LR bursts of validalition-set of track-2.
	- sr_results_track2_val_set: saved output SR images of our network for track2_val_set.
	- track2_test_set: Given LR bursts of test-set of track-2.
	- sr_results_track2_test_set: saved output SR images of our network for track2_test_set.

- ### Running Time (seconds per image):
	- Track-1 valset: 0.3237
	- Track-1 testset: 0.3350
	- Track-2 valset: 0.8861
	- Track-2 testset: 0.8838

