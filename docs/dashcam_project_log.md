# Dashcam Project Log

## checkpoints/pretrained_openpilot_1.pt模型性能
```
 python3 tools/dashcam/eval_pretrained.py \
    --model checkpoints/pretrained_openpilot_1.pt \
    --data-dir data/dual_camera_train/Town04_003 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_003
Model:   checkpoints/pretrained_openpilot_1.pt
Device:  cuda
Samples: 19998
Loading pretrained model...
  Parameters: 23,054,744
  [     1/19998]  1.2 samples/s  ETA: 16983s
  [  2000/19998]  71.5 samples/s  ETA: 252s
  [  4000/19998]  75.9 samples/s  ETA: 211s
  [  6000/19998]  75.6 samples/s  ETA: 185s
  [  8000/19998]  75.2 samples/s  ETA: 160s
  [ 10000/19998]  75.2 samples/s  ETA: 133s
  [ 12000/19998]  75.3 samples/s  ETA: 106s
  [ 14000/19998]  75.5 samples/s  ETA: 79s
  [ 16000/19998]  75.4 samples/s  ETA: 53s
  [ 18000/19998]  75.4 samples/s  ETA: 27s

Processed 19998 samples in 265.9s (75.2 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.2866     0.4635         valid=188727
lane_lines_prob                                                acc=0.9676
road_edges (≤80.0m)                1.7626     4.5719         valid=839916
lead                               2.4039     6.4617           valid=6243
lead_prob                                                      acc=0.9727
pose                               0.0728     0.3120
  tx                               0.3808
  ty                               0.0300
  tz                               0.0083
  rx                               0.0015
  ry                               0.0035
  rz                               0.0127
road_transform                     0.0156     0.0483
  tx                               0.0000
  ty                               0.0000
  tz                               0.0833
  rx                               0.0048
  ry                               0.0039
  rz                               0.0016
wide_from_device_euler             0.0044     0.0073
  roll                             0.0000
  pitch                            0.0069
  yaw                              0.0063
===========================================================================
Done.

```

## pretrained_openpilot_fp16.onnx性能
```
python3 tools/dashcam/eval_pretrained.py     --model checkpoints/pretrained_openpilot_fp16.onnx --data-dir data/dual_camera_train/Town04_003 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_003
Model:   checkpoints/pretrained_openpilot_fp16.onnx
Device:  cuda
Samples: 19998
Loading pretrained model...
  ORT provider: CUDAExecutionProvider
  Parameters: 19,645,916
  [     1/19998]  0.2 samples/s  ETA: 100252s
  [  2000/19998]  67.7 samples/s  ETA: 266s
  [  4000/19998]  74.8 samples/s  ETA: 214s
  [  6000/19998]  75.7 samples/s  ETA: 185s
  [  8000/19998]  76.0 samples/s  ETA: 158s
  [ 10000/19998]  76.3 samples/s  ETA: 131s
  [ 12000/19998]  76.7 samples/s  ETA: 104s
  [ 14000/19998]  77.1 samples/s  ETA: 78s
  [ 16000/19998]  77.3 samples/s  ETA: 52s
  [ 18000/19998]  77.3 samples/s  ETA: 26s

Processed 19998 samples in 258.8s (77.3 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.2868     0.4637         valid=188727
lane_lines_prob                                                acc=0.9677
road_edges (≤80.0m)                1.7612     4.5718         valid=839916
lead                               2.4015     6.4591           valid=6243
lead_prob                                                      acc=0.9728
pose                               0.0728     0.3121
  tx                               0.3809
  ty                               0.0300
  tz                               0.0083
  rx                               0.0015
  ry                               0.0035
  rz                               0.0127
road_transform                     0.0156     0.0483
  tx                               0.0000
  ty                               0.0000
  tz                               0.0833
  rx                               0.0048
  ry                               0.0039
  rz                               0.0016
wide_from_device_euler             0.0044     0.0073
  roll                             0.0000
  pitch                            0.0069
  yaw                              0.0063
===========================================================================
Done.
(openpilot) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/openpilot$


```


```
python3 tools/dashcam/eval_pretrained.py     --model checkpoints/pretrained_openpilot_fp16.onnx --data-dir data/dual_camera_train/Town04_004 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_004
Model:   checkpoints/pretrained_openpilot_fp16.onnx
Device:  cuda
Samples: 19998
Loading pretrained model...
  ORT provider: CUDAExecutionProvider
  Parameters: 19,645,916
  [     1/19998]  0.2 samples/s  ETA: 104595s
  [  2000/19998]  73.4 samples/s  ETA: 245s
  [  4000/19998]  79.6 samples/s  ETA: 201s
  [  6000/19998]  81.8 samples/s  ETA: 171s
  [  8000/19998]  82.9 samples/s  ETA: 145s
  [ 10000/19998]  83.2 samples/s  ETA: 120s
  [ 12000/19998]  84.9 samples/s  ETA: 94s
  [ 14000/19998]  84.8 samples/s  ETA: 71s
  [ 16000/19998]  85.1 samples/s  ETA: 47s
  [ 18000/19998]  85.3 samples/s  ETA: 23s

Processed 19998 samples in 234.1s (85.4 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.5613     1.0413         valid=105567
lane_lines_prob                                                acc=0.9734
road_edges (≤80.0m)                2.4554     5.9309         valid=839916
lead                               2.7157     9.2026           valid=2277
lead_prob                                                      acc=0.9253
pose                               0.1082     0.5612
  tx                               0.5600
  ty                               0.0396
  tz                               0.0291
  rx                               0.0018
  ry                               0.0029
  rz                               0.0158
road_transform                     0.0172     0.0484
  tx                               0.0000
  ty                               0.0000
  tz                               0.0908
  rx                               0.0072
  ry                               0.0039
  rz                               0.0013
wide_from_device_euler             0.0110     0.0169
  roll                             0.0000
  pitch                            0.0266
  yaw                              0.0063
===========================================================================
Done.
(openpilot) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/openpilot$


```

## pretrained_openpilot_fp16_sim.onnx性能
```
 python tools/dashcam/eval_pretrained.py --model checkpoints/pretrained_openpilot_fp16_sim.onnx --data-dir data/dual_camera_train/Town04_003 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_003
Model:   checkpoints/pretrained_openpilot_fp16_sim.onnx
Device:  cuda
Samples: 19998
Loading pretrained model...
  ORT provider: CUDAExecutionProvider
  Parameters: 19,645,918
  [     1/19998]  0.2 samples/s  ETA: 104383s
  [  2000/19998]  70.8 samples/s  ETA: 254s
  [  4000/19998]  78.5 samples/s  ETA: 204s
  [  6000/19998]  80.7 samples/s  ETA: 173s
  [  8000/19998]  82.3 samples/s  ETA: 146s
  [ 10000/19998]  83.1 samples/s  ETA: 120s
  [ 12000/19998]  83.8 samples/s  ETA: 95s
  [ 14000/19998]  84.0 samples/s  ETA: 71s
  [ 16000/19998]  84.6 samples/s  ETA: 47s
  [ 18000/19998]  84.8 samples/s  ETA: 24s

Processed 19998 samples in 234.9s (85.1 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.4853     0.9454         valid=959133
lane_lines_prob                                                acc=0.9001
road_edges (≤80.0m)                1.3269     1.6693         valid=839916
lead                               8.4957    30.1408            valid=458
lead_prob                                                      acc=0.9828
pose                               0.2074     1.3491
  tx                               1.0625
  ty                               0.0432
  tz                               0.1294
  rx                               0.0017
  ry                               0.0016
  rz                               0.0057
road_transform                     0.0066     0.0184
  tx                               0.0000
  ty                               0.0000
  tz                               0.0253
  rx                               0.0051
  ry                               0.0074
  rz                               0.0020
wide_from_device_euler             0.0060     0.0097
  roll                             0.0000
  pitch                            0.0090
  yaw                              0.0090
===========================================================================
Done.
(openpilot) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/openpilot$
```

```
python tools/dashcam/eval_pretrained.py --model checkpoints/pretrained_openpilot_fp16_sim.onnx --data-dir data/dual_camera_train/Town04_004 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_004
Model:   checkpoints/pretrained_openpilot_fp16_sim.onnx
Device:  cuda
Samples: 19998
Loading pretrained model...
  ORT provider: CUDAExecutionProvider
  Parameters: 19,645,918
  [     1/19998]  0.2 samples/s  ETA: 100486s
  [  2000/19998]  74.0 samples/s  ETA: 243s
  [  4000/19998]  80.0 samples/s  ETA: 200s
  [  6000/19998]  82.1 samples/s  ETA: 171s
  [  8000/19998]  84.1 samples/s  ETA: 143s
  [ 10000/19998]  84.5 samples/s  ETA: 118s
  [ 12000/19998]  85.7 samples/s  ETA: 93s
  [ 14000/19998]  85.6 samples/s  ETA: 70s
  [ 16000/19998]  85.7 samples/s  ETA: 47s
  [ 18000/19998]  86.0 samples/s  ETA: 23s

Processed 19998 samples in 232.4s (86.0 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.5613     1.0413         valid=105567
lane_lines_prob                                                acc=0.9734
road_edges (≤80.0m)                2.4554     5.9309         valid=839916
lead                               2.7157     9.2026           valid=2277
lead_prob                                                      acc=0.9253
pose                               0.1082     0.5612
  tx                               0.5600
  ty                               0.0396
  tz                               0.0291
  rx                               0.0018
  ry                               0.0029
  rz                               0.0158
road_transform                     0.0172     0.0484
  tx                               0.0000
  ty                               0.0000
  tz                               0.0908
  rx                               0.0072
  ry                               0.0039
  rz                               0.0013
wide_from_device_euler             0.0110     0.0169
  roll                             0.0000
  pitch                            0.0266
  yaw                              0.0063
===========================================================================
Done.
(openpilot) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/openpilot$
```

```
python tools/dashcam/eval_pretrained.py --model checkpoints/pretrained_openpilot_fp16_sim.onnx --data-dir data/dual_camera_train/Town04_005 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_005
Model:   checkpoints/pretrained_openpilot_fp16_sim.onnx
Device:  cuda
Samples: 9998
Loading pretrained model...
  ORT provider: CUDAExecutionProvider
  Parameters: 19,645,918
  [     1/9998]  0.2 samples/s  ETA: 57885s
  [  2000/9998]  33.3 samples/s  ETA: 240s
  [  4000/9998]  35.0 samples/s  ETA: 171s
  [  6000/9998]  35.6 samples/s  ETA: 112s
  [  8000/9998]  36.0 samples/s  ETA: 56s

Processed 9998 samples in 276.5s (36.2 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.3329     0.6933         valid=490854
lane_lines_prob                                                acc=0.9618
road_edges (≤80.0m)                0.6490     0.9225         valid=419916
lead                               1.6198     3.4027             valid=29
lead_prob                                                      acc=0.9994
pose                               0.0820     0.2673
  tx                               0.3440
  ty                               0.0229
  tz                               0.1186
  rx                               0.0013
  ry                               0.0013
  rz                               0.0039
road_transform                     0.0051     0.0164
  tx                               0.0000
  ty                               0.0000
  tz                               0.0169
  rx                               0.0051
  ry                               0.0069
  rz                               0.0019
wide_from_device_euler             0.0020     0.0031
  roll                             0.0000
  pitch                            0.0036
  yaw                              0.0023
===========================================================================
Done.
(openpilot) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/openpilot$
```

```
python tools/dashcam/eval_pretrained.py --model checkpoints/pretrained_openpilot_fp16_sim.onnx --data-dir data/dual_camera_train/Town04_004 --max-dist 80
Max dist: 80.0m  -> using 21/33 points (last x=75.0m)
Dataset: data/dual_camera_train/Town04_004
Model:   checkpoints/pretrained_openpilot_fp16_sim.onnx
Device:  cuda
Samples: 19998
Loading pretrained model...
  ORT provider: CUDAExecutionProvider
  Parameters: 19,645,918
  [     1/19998]  0.2 samples/s  ETA: 107970s
  [  2000/19998]  33.9 samples/s  ETA: 531s
  [  4000/19998]  35.2 samples/s  ETA: 455s
  [  6000/19998]  35.7 samples/s  ETA: 392s
  [  8000/19998]  36.0 samples/s  ETA: 333s
  [ 10000/19998]  36.1 samples/s  ETA: 277s
  [ 12000/19998]  36.3 samples/s  ETA: 220s
  [ 14000/19998]  36.4 samples/s  ETA: 165s
  [ 16000/19998]  36.5 samples/s  ETA: 110s
  [ 18000/19998]  36.5 samples/s  ETA: 55s

Processed 19998 samples in 547.1s (36.6 samples/s)

===========================================================================
Component                             MAE       RMSE                Extra
---------------------------------------------------------------------------
lane_lines (≤80.0m)                0.4056     0.8159         valid=105567
lane_lines_prob                                                acc=0.9812
road_edges (≤80.0m)                1.4487     3.8574         valid=839916
lead                               1.2892     4.2675           valid=2277
lead_prob                                                      acc=0.9916
pose                               0.0576     0.2891
  tx                               0.2886
  ty                               0.0219
  tz                               0.0227
  rx                               0.0015
  ry                               0.0023
  rz                               0.0086
road_transform                     0.0092     0.0247
  tx                               0.0000
  ty                               0.0000
  tz                               0.0455
  rx                               0.0055
  ry                               0.0028
  rz                               0.0015
wide_from_device_euler             0.0114     0.0173
  roll                             0.0000
  pitch                            0.0227
  yaw                              0.0116
===========================================================================
Done.
(openpilot) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/openpilot$
```