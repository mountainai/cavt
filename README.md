## Video Transformer for Engagement Prediction Task - PyTorch
Implementation of the Class Attention in Video Transformer

### Usage
For space limitation,send two CavT model files trained on the EmotiW-EP dataset and DAiSEE dataset upon request.
Then put them under optimal_checkpoint directory.

Step 1: run OpenFace [https://github.com/TadasBaltrusaitis/OpenFace/wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki) 
   with Docker to perform face extraction on multiple videos.

FeatureExtraction.exe -nomask -f "video1.avi" -f "video2.avi" -f "video3.avi"

Step 2: Run tools/test.sh to reproduce our results for two CavT models as follows.

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
