# Third-Party Software Licenses & Notices

The NEXORA CUDA GPU Acceleration Engine incorporates, links against, depends upon, or was built using third-party open-source software libraries. This document provides attribution notices, copyright statements, license designations, and governing terms required by the respective licensors.

> [!NOTE]
> **Distribution Scope Clarification:** The NEXORA official source distribution archive (`cuda-acceleration-engine-v1.0.0-rc4-source.zip`) contains 100% first-party authored source code. Third-party runtime packages (such as PyTorch, NumPy, Uvicorn, React) are **not bundled** in the source archive; they are installed out-of-band by end-users via package managers (PyPI, npm). This document is provided in compliance with open-source attribution requirements and supply-chain transparency.

---

## Table of Contents
1. [NVIDIA CUDA Toolkit Acknowledgement](#1-nvidia-cuda-toolkit-acknowledgement)
2. [Master Third-Party Dependency Inventory](#2-master-third-party-dependency-inventory)
   - [2.1 Python Runtime & Backend Environment (.venv)](#21-python-runtime--backend-environment-venv)
   - [2.2 Frontend Web App Environment (frontend/)](#22-frontend-web-app-environment-frontend)
3. [Deep-Dive Attributions & Subcomponent Notices](#3-deep-dive-attributions--subcomponent-notices)
   - [3.1 PyTorch (torch 2.6.0+cu124)](#31-pytorch-torch-260cu124)
   - [3.2 NumPy (numpy 2.4.6)](#32-numpy-numpy-246)
   - [3.3 Requests (requests 2.34.2)](#33-requests-requests-2342)
   - [3.4 Certifi (certifi 2026.7.22)](#34-certifi-certifi-2026722)
   - [3.5 FastAPI & Pydantic](#35-fastapi--pydantic)
   - [3.6 Uvicorn & Starlette](#36-uvicorn--starlette)
   - [3.7 Packaging (packaging 26.3)](#37-packaging-packaging-263)
   - [3.8 TypeScript & Frontend Tooling](#38-typescript--frontend-tooling)
4. [Verbatim License Texts](#4-verbatim-license-texts)
   - [4.1 MIT License](#41-mit-license)
   - [4.2 BSD 3-Clause License](#42-bsd-3-clause-license)
   - [4.3 BSD 2-Clause License](#43-bsd-2-clause-license)
   - [4.4 Apache License, Version 2.0](#44-apache-license-version-20)
   - [4.5 Mozilla Public License, Version 2.0 (MPL-2.0)](#45-mozilla-public-license-version-20-mpl-20)
   - [4.6 Python Software Foundation License Version 2.0 (PSF-2.0)](#46-python-software-foundation-license-version-20-psf-20)
   - [4.7 ISC License](#47-isc-license)
5. [Status of Unverified Rights & Disclosures](#5-status-of-unverified-rights--disclosures)

---

## 1. NVIDIA CUDA Toolkit Acknowledgement

Portions of this software are based on NVIDIA CUDA Toolkit components.

NEXORA is independently developed and is neither sponsored, endorsed, nor certified by NVIDIA Corporation.
NVIDIA, CUDA, and GeForce RTX are trademarks or registered trademarks of NVIDIA Corporation in the United States and other countries.

---

## 2. Master Third-Party Dependency Inventory

### 2.1 Python Runtime & Backend Environment (.venv)

| Package Name | Exact Version | Classification | SPDX License Identifier | Copyright Holders / Authors | Upstream Notice Obligation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `annotated-doc` | 0.0.5 | Transitive | MIT | Sebastián Ramírez (2025) | Retain MIT copyright notice |
| `annotated-types` | 0.8.0 | Transitive | MIT | annotated-types contributors (2022) | Retain MIT copyright notice |
| `anyio` | 4.15.1 | Transitive | MIT | Alex Grönholm (2018) | Retain MIT copyright notice |
| `certifi` | 2026.7.22 | Transitive | MPL-2.0 | Mozilla NSS / Kenneth Reitz | MPL-2.0 § 3.3 Larger Works notice |
| `charset-normalizer` | 3.5.2 | Transitive | MIT | TAHRI Ahmed R. (2025) | Retain MIT copyright notice |
| `click` | 8.5.0 | Transitive | BSD-3-Clause | Pallets (2014) | Retain BSD-3 notice & disclaimer |
| `colorama` | 0.4.6 | Transitive | BSD-3-Clause | Jonathan Hartley (2010) | Retain BSD-3 notice & disclaimer |
| `fastapi` | 0.146.0 | **Direct** | MIT | Sebastián Ramírez (2018) | Retain MIT copyright notice |
| `filelock` | 3.32.3 | Transitive | MIT | Bernát Gábor and contributors (2025) | Retain MIT copyright notice |
| `fsspec` | 2026.7.0 | Transitive | BSD-3-Clause | Martin Durant (2018) | Retain BSD-3 notice & disclaimer |
| `h11` | 0.16.0 | Transitive | MIT | Nathaniel J. Smith (2016) | Retain MIT copyright notice |
| `httpcore` | 1.0.9 | Transitive | BSD-3-Clause | Encode OSS Ltd (2020) | Retain BSD-3 notice & disclaimer |
| `httpx` | 0.28.1 | **Direct** | BSD-3-Clause | Encode OSS Ltd (2019) | Retain BSD-3 notice & disclaimer |
| `idna` | 3.20 | Transitive | BSD-3-Clause | Kim Davies and contributors (2013-2026) | Retain BSD-3 notice & disclaimer |
| `iniconfig` | 2.3.1 | Transitive | MIT | Holger Krekel (2010-2023) | Retain MIT copyright notice |
| `Jinja2` | 3.1.6 | Transitive | BSD-3-Clause | Pallets (2007) | Retain BSD-3 notice & disclaimer |
| `MarkupSafe` | 3.0.3 | Transitive | BSD-3-Clause | Pallets (2010) | Retain BSD-3 notice & disclaimer |
| `mpmath` | 1.3.0 | Transitive | BSD-3-Clause | Fredrik Johansson (2005-2021) | Retain BSD-3 notice & disclaimer |
| `networkx` | 3.6.1 | Transitive | BSD-3-Clause | NetworkX Developers (2004-2025) | Retain BSD-3 notice & disclaimer |
| `numpy` | 2.4.6 | **Direct** | BSD-3-Clause (+14 subcomponents) | NumPy Developers (2005-2025) et al. | Retain BSD-3 notice & subcomponent texts |
| `opentelemetry-api` | 1.45.1 | Transitive | Apache-2.0 | OpenTelemetry Authors | Apache-2.0 § 4(d) Attribution retention |
| `packaging` | 26.3 | Transitive | BSD-2-Clause / Apache-2.0 | Donald Stufft and individual contributors | Elect BSD-2-Clause; retain notice |
| `pip` | 24.0 | Transitive | MIT | The pip developers (2008-present) | Retain MIT copyright notice |
| `pluggy` | 1.6.0 | Transitive | MIT | Holger Krekel (2015) | Retain MIT copyright notice |
| `pydantic` | 2.14.1 | **Direct** | MIT | Pydantic Services Inc. (2017-present) | Retain MIT copyright notice |
| `pydantic_core` | 2.50.1 | Transitive | MIT | Samuel Colvin (2022) | Retain MIT copyright notice |
| `Pygments` | 2.21.0 | Transitive | BSD-2-Clause | Pygments Team (2006-2022) | Retain BSD-2 notice & disclaimer |
| `pytest` | 9.1.1 | **Direct** | MIT | Holger Krekel and others (2004) | Retain MIT copyright notice |
| `PyYAML` | 6.0.3 | **Direct** | MIT | Ingy döt Net / Kirill Simonov (2017) | Retain MIT copyright notice |
| `requests` | 2.34.2 | **Direct** | Apache-2.0 | Kenneth Reitz (2019) | Apache-2.0 § 4(d) Mandatory NOTICE retention |
| `setuptools` | 65.5.0 | Transitive | MIT | Jason R. Coombs | Retain MIT copyright notice |
| `starlette` | 1.9.0 | **Direct** | BSD-3-Clause | Encode OSS Ltd (2018) | Retain BSD-3 notice & disclaimer |
| `sympy` | 1.13.1 | Transitive | BSD-3-Clause | SymPy Development Team (2006-2023) | Retain BSD-3 notice & disclaimer |
| `torch` | 2.6.0+cu124 | **Direct** | BSD-3-Clause (PyTorch) | Meta Platforms, Idiap, DeepMind et al. | Retain PyTorch BSD notice, sub-notices, & NOTICE |
| `typing-inspection` | 0.4.4 | Transitive | MIT | Pydantic Services Inc. (2024) | Retain MIT copyright notice |
| `typing_extensions` | 4.16.0 | Transitive | PSF-2.0 / 0BSD | Python Software Foundation | Retain PSF License Agreement & notices |
| `urllib3` | 2.8.0 | Transitive | MIT | Andrey Petrov and contributors (2008) | Retain MIT copyright notice |
| `uvicorn` | 0.54.0 | **Direct** | BSD-3-Clause | Encode OSS Ltd (2017-present) | Retain BSD-3 notice & disclaimer |

### 2.2 Frontend Web App Environment (frontend/)

| Package Name | Exact Version | Scope | SPDX License Identifier | Copyright Holders / Authors | Upstream Notice Obligation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clsx` | 2.1.1 | Production | MIT | Luke Edwards | Retain MIT copyright notice |
| `lucide-react` | 0.475.0 | Production | ISC | Lucide Contributors | Retain ISC copyright notice |
| `react` | 18.3.1 | Production | MIT | Meta Platforms, Inc. | Retain MIT copyright notice |
| `react-dom` | 18.3.1 | Production | MIT | Meta Platforms, Inc. | Retain MIT copyright notice |
| `recharts` | 2.15.4 | Production | MIT | Recharts Group | Retain MIT copyright notice |
| `tailwind-merge` | 3.7.0 | Production | MIT | Dany Castillo | Retain MIT copyright notice |
| `@types/node` | 22.20.5 | Dev / Build | MIT | DefinitelyTyped Contributors | Retain MIT copyright notice |
| `@types/react` | 18.3.31 | Dev / Build | MIT | DefinitelyTyped Contributors | Retain MIT copyright notice |
| `@types/react-dom` | 18.3.7 | Dev / Build | MIT | DefinitelyTyped Contributors | Retain MIT copyright notice |
| `@vitejs/plugin-react` | 4.7.0 | Dev / Build | MIT | Vite Contributors | Retain MIT copyright notice |
| `autoprefixer` | 10.6.1 | Dev / Build | MIT | Andrey Sitnik | Retain MIT copyright notice |
| `postcss` | 8.5.30 | Dev / Build | MIT | Andrey Sitnik | Retain MIT copyright notice |
| `tailwindcss` | 3.4.19 | Dev / Build | MIT | Tailwind Labs, Inc. | Retain MIT copyright notice |
| `typescript` | 5.9.3 | Dev / Build | Apache-2.0 | Microsoft Corporation | Retain Apache-2.0 & ThirdPartyNotices |
| `vite` | 6.4.4 | Dev / Build | MIT | Yuxi (Evan) You & Vite Contributors | Retain MIT copyright notice |

---

## 3. Deep-Dive Attributions & Subcomponent Notices

### 3.1 PyTorch (torch 2.6.0+cu124)
- **License:** BSD-3-Clause (PyTorch License with Facebook Copyright Grant)
- **Copyright:**
  ```text
  From PyTorch:
  Copyright (c) 2016-present Facebook, Inc            (Adam Paszke)
  Copyright (c) 2014-present Facebook, Inc            (Soumith Chintala)
  Copyright (c) 2011-2014 Idiap Research Institute    (Ronan Collobert)
  Copyright (c) 2012-2014 Deepmind Technologies       (Koray Kavukcuoglu)
  Copyright (c) 2011-2012 NEC Laboratories America    (Koray Kavukcuoglu)
  Copyright (c) 2011-2013 NYU                         (Clement Farabet)
  Copyright (c) 2006-2010 NEC Laboratories America    (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
  Copyright (c) 2006      Idiap Research Institute    (Samy Bengio)
  Copyright (c) 2001-2004 Idiap Research Institute    (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

  From Caffe2:
  Copyright (c) 2016-present, Facebook Inc. All rights reserved.
  Copyright (c) 2015 Google Inc. All rights reserved.
  Copyright (c) 2015 Yangqing Jia. All rights reserved.
  Copyright 2019-2020 Kakao Brain.
  ```
- **Redistribution Terms:** Redistribution and use in source and binary forms, with or without modification, are permitted provided that the copyright notice, list of conditions, and disclaimer are retained.

### 3.2 NumPy (numpy 2.4.6)
- **License:** BSD-3-Clause
- **Copyright:** Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
- **Embedded Subcomponents (licenses/ directory):**
  1. **PocketFFT:** Copyright (C) 2010-2018 Max-Planck-Society (BSD-3-Clause)
  2. **LAPACK Lite:** Copyright (c) 1992-2013 The University of Tennessee (BSD-3-Clause)
  3. **Masked Arrays (ma):** Copyright (c) 2006, University of Georgia (BSD-3-Clause)
  4. **Random Module:** Dual-licensed under UI/NCSA Open Source License and BSD-3-Clause
  5. **MT19937:** Copyright (c) 2003-2005, Jean-Sebastien Roy (BSD-style)
  6. **PCG64:** Copyright 2014 Melissa O'Neill (MIT)
  7. **Philox:** Copyright 2010-2012, D. E. Shaw Research (BSD-3-Clause)
  8. **SFC64:** Adapted from Chris Doty-Humphrey (MIT)
  9. **SplitMix64:** Written in 2015 by Sebastiano Vigna (CC0 1.0 Universal / Public Domain)
  10. **libdivide:** Copyright (C) 2010-2019 ridiculous_fish (Zlib License)
  11. **pythoncapi-compat:** Copyright Contributors to pythoncapi_compat (0BSD License)
  12. **highway:** Dual-licensed under Apache License 2.0 or BSD-3-Clause
  13. **dragon4:** Copyright (c) 2014 Ryan Juckett (MIT)
  14. **x86-simd-sort:** Copyright (c) 2022, Intel Corporation (BSD-3-Clause)

### 3.3 Requests (requests 2.34.2)
- **License:** Apache License, Version 2.0
- **Copyright & Mandatory NOTICE:**
  ```text
  Requests
  Copyright 2019 Kenneth Reitz
  ```

### 3.4 Certifi (certifi 2026.7.22)
- **License:** Mozilla Public License, Version 2.0 (MPL-2.0)
- **Notice:** This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

### 3.5 FastAPI & Pydantic
- **FastAPI:** Copyright (c) 2018 Sebastián Ramírez (MIT)
- **Pydantic:** Copyright (c) 2017-present Pydantic Services Inc. (MIT)

### 3.6 Uvicorn & Starlette
- **Uvicorn:** Copyright (c) 2017-present Encode OSS Ltd (BSD-3-Clause)
- **Starlette:** Copyright (c) 2018 Encode OSS Ltd (BSD-3-Clause)

### 3.7 Packaging (packaging 26.3)
- **Dual License Election:** Disjunctive dual-licensed under Apache-2.0 OR BSD-2-Clause.
- **NEXORA Election:** NEXORA formally elects the **BSD-2-Clause** license to eliminate downstream Apache patent retaliation ambiguities.
- **Copyright:** Copyright (c) Donald Stufft and individual contributors.

### 3.8 TypeScript & Frontend Tooling
- **TypeScript:** Copyright (c) Microsoft Corporation (Apache-2.0)
- **Lucide React:** Copyright (c) Lucide Contributors (ISC)
- **React & React DOM:** Copyright (c) Meta Platforms, Inc. (MIT)
- **Vite:** Copyright (c) 2019-present Yuxi (Evan) You & Vite Contributors (MIT)

---

## 4. Verbatim License Texts

### 4.1 MIT License
```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 4.2 BSD 3-Clause License
```text
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

### 4.3 BSD 2-Clause License
```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

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

### 4.4 Apache License, Version 2.0
```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### 4.5 Mozilla Public License, Version 2.0 (MPL-2.0)
```text
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

### 4.6 Python Software Foundation License Version 2.0 (PSF-2.0)
```text
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------
1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
the Individual or Organization ("Licensee") accessing and otherwise using this
software ("Python") in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works, distribute,
and otherwise use Python alone or in any derivative version, provided, however, that
PSF's License Agreement and PSF's notice of copyright, i.e., "Copyright (c) 2001-2026
Python Software Foundation; All Rights Reserved" are retained in Python alone or in
any derivative version prepared by Licensee.
```

### 4.7 ISC License
```text
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

---

## 5. Status of Unverified Rights & Disclosures

In strict adherence to release compliance standards, the following areas are formally designated as **UNVERIFIED / PENDING**:

1. **Model Checkpoint Weights Redistribution:**
   - **Status:** **UNVERIFIED / HOSTING PAUSED**.
   - Model parameter tensors (`*.pt`) do not automatically inherit MIT software terms. Public redistribution on external hubs (GitHub Releases, Hugging Face) remains paused until the Project Owner executes the licensing framework decision in `MODEL_CARD.md` Section 9.
2. **Contributor Title Affirmation Execution:**
   - **Status:** **UNVERIFIED / PENDING FORMAL EXECUTION**.
   - While original authorship is affirmed, formal signed DCO / Non-Exclusive Licensing instruments have not been executed by all individual contributors.
