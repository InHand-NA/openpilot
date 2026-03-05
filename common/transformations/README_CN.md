
# 参考坐标系
------
代码库中使用了多种参考坐标系。本目录包含在各坐标系之间进行变换所需的全部辅助函数。变换通常通过生成旋转矩阵并做矩阵乘法来完成。


| 名称 | [x, y, z] | 单位 | 说明 |
| :-------------: |:-------------:| :-----:| :----: |
| Geodetic（大地坐标系） | [纬度, 经度, 高度] | 大地坐标 | 有时以 [经度, 纬度, 高度] 顺序使用，建议避免使用此坐标系。 |
| ECEF（地心地固坐标系） | [x, y, z] | 米 | 使用 **ITRF14 (IGS14)**，而非 NAD83。<br>这是 Mesh3D 的全局坐标系。 |
| NED（北东地坐标系） | [北, 东, 下] | 米 | 相对于地球表面，适合可视化使用。 |
| Device（设备帧） | [前, 右, 下] | 米 | Mesh3D 的局部坐标系。<br>以**相机**为原点，**非 IMU**。<br> ![img](http://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/RPY_angles_of_airplanes.png/440px-RPY_angles_of_airplanes.png)|
| Calibrated（标定帧） | [前, 右, 下] | 米 | **模型输出所在的坐标系**。<br>详见下文。|
| Car（车辆帧坐标系） | [前, 右, 下] | 米 | 适合估算道路上各点位置。<br>详见下文。|
| View（视图帧坐标系） | [右, 下, 前] | 米 | 与设备帧相同，但遵循相机坐标约定。 |
| Camera（相机帧坐标系） | [u, v, 焦距] | 像素 | 与视图帧类似，但为相机图像上的二维坐标。|
| Normalized Camera（归一化相机帧） | [u / 焦距, v / 焦距, 1] | / | |
| Model（模型帧坐标系） | [u, v, 焦距] | 像素 | 模型所使用的、从完整相机图像中裁取的矩形区域。 |
| Normalized Model（归一化模型帧坐标系） | [u / 焦距, v / 焦距, 1] | / | |




# 姿态表示约定
------
四元数、旋转矩阵和欧拉角是姿态的三种等价表示方式，代码库中三种均有使用。

欧拉角优先采用 [roll, pitch, yaw]（滚转、俯仰、偏航）的约定，分别对应绕 [x, y, z] 轴的旋转。所有欧拉角均应使用弧度或弧度/秒，仅在绘图或显示时例外。四元数采用 Hamilton 表示法，格式为 [q<sub>w</sub>, q<sub>x</sub>, q<sub>y</sub>, q<sub>z</sub>]，所有四元数必须归一化且 q<sub>w</sub> 严格为正。**四元数是姿态的唯一表示，而欧拉角和旋转矩阵不具备此性质。**

使用欧拉角进行坐标系间旋转时，约定依次绕 roll、pitch、yaw 旋转，且每次均绕**旋转后的轴**（而非原始轴）旋转。


# 车辆帧坐标系（Car frame）
------
设备帧坐标系与 openpilot 所用的前向摄像头对齐。然而，**在控制车辆时，使用与车辆对齐的参考坐标系更为便利**，而这两个坐标系可能并不相同。

**车辆帧坐标系的方向定义**为：当车辆在平坦道路上直行时，与车辆行驶方向及路面平面对齐。车辆帧的原点定义为设备帧坐标系正下方（在车辆帧中）、位于路面平面上的点。由于悬架运动等因素，该坐标系的位置和方向不一定始终与行驶方向或路面平面保持对齐。


# 标定帧坐标系（Calibrated frame）
------
为使 openpilot 驾驶模型在不同车辆、不同安装方式下接收到视觉上相似的输入图像，需要将图像"标定"变换到标定帧坐标系中。标定帧坐标系在俯仰（pitch）和偏航（yaw）方向上与车辆帧坐标系对齐，在滚转（roll）方向上与设备帧对齐，且与设备帧共享同一原点。


# 示例
------
将全局 Mesh3D 的位置和姿态（positions_ecef、quats_ecef）变换到以第一个 Mesh3D 位置和姿态描述的局部坐标系中：

```python
ecef_from_local = rot_from_quat(quats_ecef[0])
local_from_ecef = ecef_from_local.T
positions_local = np.einsum('ij,kj->ki', local_from_ecef, postions_ecef - positions_ecef[0])
rotations_global = rot_from_quat(quats_ecef)
rotations_local = np.einsum('ij,kjl->kil', local_from_ecef, rotations_global)
eulers_local = euler_from_rot(rotations_local)
```
