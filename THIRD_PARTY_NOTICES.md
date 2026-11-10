# Third-Party Notices

This repository is released under the Apache License 2.0 (see `LICENSE`). It includes, adapts, or
references the third-party material listed below, which remains under its own terms. Dependencies
that are installed separately (`requirements.txt`) are not redistributed and keep their own licenses.

## Adapted code

These files re-implement or adapt parts of upstream projects for this repository's data pipeline.
The license texts of the BSD projects are reproduced at the end of this file.

| Files | Upstream | License |
|---|---|---|
| `src/model/airimu.py` | [AirIMU](https://github.com/haleqiu/AirIMU) | BSD 3-Clause, Copyright (c) 2024, Carnegie Mellon University |
| `src/eval_airio_ekf.py`, `src/model/airio.py` | [AirIO](https://github.com/Air-IO/Air-IO) | BSD 3-Clause, Copyright (c) 2025, Carnegie Mellon University |
| `src/model/tlio.py` | [TLIO](https://github.com/CathIAS/TLIO) | BSD, Copyright (c) Facebook, Inc. and its affiliates |
| `src/model/tartanimu.py` | [TartanIMU](https://github.com/superxslam/TartanIMU) | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

## Git submodules

These projects are referenced as git submodules. Their code is not part of this repository, and
each checkout carries its own license file.

| Path | Project | License |
|---|---|---|
| `thirdparty/airimu` | [AirIMU](https://github.com/haleqiu/AirIMU) | BSD 3-Clause |
| `thirdparty/airio` | [AirIO](https://github.com/Air-IO/Air-IO) | BSD 3-Clause |
| `thirdparty/tlio` | [TLIO](https://github.com/CathIAS/TLIO) | BSD |
| `thirdparty/tartanimu` | [TartanIMU](https://github.com/superxslam/TartanIMU) | Apache 2.0 |
| `thirdparty/imagebind` | [ImageBind](https://github.com/facebookresearch/ImageBind) | CC BY-NC-SA 4.0 |
| `thirdparty/imu2clip` | [IMU2CLIP](https://github.com/facebookresearch/imu2clip) | CC BY-NC-SA 4.0 |
| `assets/unitree` | [unitree_ros](https://github.com/unitreerobotics/unitree_ros) | BSD 3-Clause |

ImageBind and IMU2CLIP are **non-commercial**. The ImageBind and IMU2CLIP baselines
(`src/model/imagebind.py`, `src/model/imu2clip.py`) import those submodules when they run, so using
them is subject to those terms. Pretrained baseline weights are not included; they are obtained
from their upstream projects under the upstream terms.

## Datasets

No dataset is included. The stem and split listings under `src/config/data/` name recordings of the
datasets below, which are obtained from their providers under their own terms.

- [Nymeria](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/dataset_explorer)
- [Virginia Tech Natural Motion](https://data.lib.vt.edu/articles/dataset/Virginia_Tech_Natural_Motion_Dataset/14114054)
- [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
- [Real G1 (humanoid-g1-odometry)](https://huggingface.co/datasets/Open-IO/humanoid-g1-odometry)

## License texts

### AirIMU and AirIO

The two projects use the same BSD 3-Clause text and differ only in the copyright year shown above.

```text
BSD 3-Clause License

Copyright (c) 2025, Carnegie Mellon University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### TLIO

```text
BSD License

For TLIO training and inference software

Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.```
