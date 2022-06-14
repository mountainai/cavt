## Video Transformer for Engagement Prediction Task - PyTorch
Implementation of the Class Attention in Video Transformer

### Usage
Step 1: Download model files from pan.baidu.com and put them optimal_checkpoint folder.

URL: https://pan.baidu.com/s/15ZxgAKGKFQEsCN-GlzKJSg

Code: cbhd

Step 2: run OpenFace [https://github.com/TadasBaltrusaitis/OpenFace/wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki) 
   with Docker to perform face extraction on multiple videos.

FeatureExtraction.exe -nomask -simscale 1  -f "video1.avi" -f "video2.avi" -f "video3.avi"

Step 3: Organize datasets according to directory structure.

```
Engwild2020
	Train_112_scale1_frames
		1100011002_aligned
			frame_det_00_000001.bmp
			frame_det_00_000001.bmp
			......
		......
	validation_112_scale1_frames
		2100011002_aligned
			frame_det_00_000001.bmp
			frame_det_00_000001.bmp
			......
		......
DAiSEE
    DataSet
		Train_112_scale1_frames
			1100011001_aligned
			    frame_det_00_000001.bmp
				frame_det_00_000001.bmp
				......
			......
		validation_112_scale1_frames
		    2100011001_aligned
			    frame_det_00_000001.bmp
				frame_det_00_000001.bmp
				......
			......
	    test_112_scale1_frames
		    3100011001_aligned
			    frame_det_00_000001.bmp
				frame_det_00_000001.bmp
				......
			......
```

Step 4: Run tools/test.sh to reproduce our results for two CavT models as follows.

```
Evaluating mse ...
mse for 0.0: 0.1107
mse for 0.33: 0.0983
mse for 0.66: 0.0119
mse for 1.0: 0.0482
Mean class mse: 0.0673
mean square error	0.0495
mse: 0.0495
Evaluating mse ...
mse for 0.0: 0.5535
mse for 0.33: 0.1770
mse for 0.66: 0.0203
mse for 1.0: 0.0396
Mean class mse: 0.1976
mean square error	0.0377
mse: 0.0377
```

### References
Some part of the code is adapted from the VideoSwinTransformer repository [https://github.com/SwinTransformer/Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).
Special thanks for their great work!
