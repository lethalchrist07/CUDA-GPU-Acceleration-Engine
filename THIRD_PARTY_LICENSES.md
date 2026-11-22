# Third-Party Software Licenses & Notices

The NEXORA CUDA GPU Acceleration Engine incorporates, links against, depends upon, or was built using third-party open-source software libraries. This document provides attribution notices, copyright statements, license designations, and governing terms required by the respective licensors for **NEXORA v1.0.0 (Release Candidate 4)**.

> [!NOTE]
> **Distribution Scope Clarification:** The NEXORA official source distribution archive (`cuda-acceleration-engine-v1.0.0-rc4-source.zip`) contains first-party authored engine and backend source code, accompanied by pre-compiled static frontend web UI distribution assets (`frontend/dist/`). Python backend dependencies (such as PyTorch, NumPy, FastAPI, Uvicorn) and system GPU runtimes (NVIDIA CUDA Toolkit, cuBLAS, cuDNN) are **not bundled** and are installed out-of-band by end-users via package managers (PyPI) and system installers. In contrast, frontend web UI runtime dependencies (including React, ReactDOM, Lucide-React, Recharts, and transitive support modules) are compiled and statically packaged within `frontend/dist/assets/index-BhQcNE8G.js` (856,464 bytes).
>
> **Release Archive Lineage Distinction:** The certified RC4 release candidate archive (`cuda-acceleration-engine-v1.0.0-rc4-source.zip`, SHA-256 `DBDD670654F15955FEF46F9CEF9EB5E8E5CDCDD7E2ECB407C566C28EDECB5AF0`) was cryptographically frozen during pre-remediation checkpointing and preserves the pre-remediation artifact state. This remediated document (`THIRD_PARTY_LICENSES.md`) represents the active, verified licensing specification prepared to accompany all future official production release distribution packages (e.g. v1.0.0 Final GA or subsequent release archives).

---

## Table of Contents
1. [NVIDIA CUDA Toolkit Acknowledgement](#1-nvidia-cuda-toolkit-acknowledgement)
2. [Master Third-Party Dependency Inventory](#2-master-third-party-dependency-inventory)
   - [2.1 Python Runtime & Backend Environment (.venv)](#21-python-runtime--backend-environment-venv)
   - [2.2 Frontend Web App Production Environment (46 Packages)](#22-frontend-web-app-production-environment-46-packages)
   - [2.3 Frontend Development & Build Tooling Environment (181 Packages)](#23-frontend-development--build-tooling-environment-181-packages)
3. [Deep-Dive Attributions & Subcomponent Notices](#3-deep-dive-attributions--subcomponent-notices)
   - [3.1 PyTorch (torch 2.6.0+cu124)](#31-pytorch-torch-260cu124)
   - [3.2 NumPy (numpy 2.4.6)](#32-numpy-numpy-246)
   - [3.3 Requests (requests 2.34.2)](#33-requests-requests-2342)
   - [3.4 Certifi (certifi 2026.7.22)](#34-certifi-certifi-2026722)
   - [3.5 FastAPI & Pydantic](#35-fastapi--pydantic)
   - [3.6 Uvicorn & Starlette](#36-uvicorn--starlette)
   - [3.7 Packaging (packaging 26.3)](#37-packaging-packaging-263)
   - [3.8 React & React-DOM (Bundled Web UI)](#38-react--react-dom-bundled-web-ui)
   - [3.9 Lucide React (Bundled Web UI)](#39-lucide-react-bundled-web-ui)
   - [3.10 Recharts & Transitive D3 Visualization Subcomponents](#310-recharts--transitive-d3-visualization-subcomponents)
   - [3.11 Frontend Development Tooling & Compilers](#311-frontend-development-tooling--compilers)
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

### 2.2 Frontend Web App Production Environment (46 Packages)

This inventory comprehensively accounts for all **46 production packages** (6 direct, 40 transitive) resolved in `frontend/package-lock.json`. These packages constitute the runtime dependency graph compiled and statically bundled into `frontend/dist/assets/index-BhQcNE8G.js`.

| Index | Package Name | Exact Version | Scope | SPDX License | Copyright Holder / Author | Upstream Notice Obligation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `@babel/runtime` | 7.29.10 | Transitive | MIT | Copyright (c) 2014-present Sebastian McKenzie and ot... | Retain notice & license text |
| 02 | `@types/d3-array` | 3.2.2 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 03 | `@types/d3-color` | 3.1.3 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 04 | `@types/d3-ease` | 3.0.2 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 05 | `@types/d3-interpolate` | 3.0.4 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 06 | `@types/d3-path` | 3.1.1 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 07 | `@types/d3-scale` | 4.0.9 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 08 | `@types/d3-shape` | 3.2.0 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 09 | `@types/d3-time` | 3.0.4 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 10 | `@types/d3-timer` | 3.0.2 | Transitive | MIT | Copyright (c) Microsoft Corporation. | Retain notice & license text |
| 11 | `clsx` | 2.1.1 | **Direct** | MIT | Copyright (c) Luke Edwards <luke.edwards05@gmail.com... | Retain notice & license text |
| 12 | `csstype` | 3.2.3 | Transitive | MIT | Copyright (c) 2017-2018 Fredrik Nicol | Retain notice & license text |
| 13 | `d3-array` | 3.2.4 | Transitive | ISC | Copyright 2010-2023 Mike Bostock | Retain notice & license text |
| 14 | `d3-color` | 3.1.0 | Transitive | ISC | Copyright 2010-2022 Mike Bostock | Retain notice & license text |
| 15 | `d3-ease` | 3.0.1 | Transitive | BSD-3-Clause | Copyright 2010-2021 Mike Bostock | Retain notice & license text |
| 16 | `d3-format` | 3.1.2 | Transitive | ISC | Copyright 2010-2026 Mike Bostock | Retain notice & license text |
| 17 | `d3-interpolate` | 3.0.1 | Transitive | ISC | Copyright 2010-2021 Mike Bostock | Retain notice & license text |
| 18 | `d3-path` | 3.1.0 | Transitive | ISC | Copyright 2015-2022 Mike Bostock | Retain notice & license text |
| 19 | `d3-scale` | 4.0.2 | Transitive | ISC | Copyright 2010-2021 Mike Bostock | Retain notice & license text |
| 20 | `d3-shape` | 3.2.0 | Transitive | ISC | Copyright 2010-2022 Mike Bostock | Retain notice & license text |
| 21 | `d3-time` | 3.1.0 | Transitive | ISC | Copyright 2010-2022 Mike Bostock | Retain notice & license text |
| 22 | `d3-time-format` | 4.1.0 | Transitive | ISC | Copyright 2010-2021 Mike Bostock | Retain notice & license text |
| 23 | `d3-timer` | 3.0.1 | Transitive | ISC | Copyright 2010-2021 Mike Bostock | Retain notice & license text |
| 24 | `decimal.js-light` | 2.5.1 | Transitive | MIT | Copyright (c) 2020 Michael Mclaughlin | Retain notice & license text |
| 25 | `dom-helpers` | 5.2.1 | Transitive | MIT | Copyright (c) 2015 Jason Quense | Retain notice & license text |
| 26 | `eventemitter3` | 4.0.7 | Transitive | MIT | Copyright (c) 2014 Arnout Kazemier | Retain notice & license text |
| 27 | `fast-equals` | 5.4.5 | Transitive | MIT | Copyright (c) 2025 Tony Quetano | Retain notice & license text |
| 28 | `internmap` | 2.0.3 | Transitive | ISC | Copyright 2021 Mike Bostock | Retain notice & license text |
| 29 | `js-tokens` | 4.0.0 | Transitive | MIT | Copyright (c) 2014, 2015, 2016, 2017, 2018 Simon Lydell | Retain notice & license text |
| 30 | `lodash` | 4.18.1 | Transitive | MIT | Copyright OpenJS Foundation and other contributors <... | Retain notice & license text |
| 31 | `loose-envify` | 1.4.0 | Transitive | MIT | Copyright (c) 2015 Andres Suarez <zertosh@gmail.com> | Retain notice & license text |
| 32 | `lucide-react` | 0.475.0 | **Direct** | ISC | Copyright (c) for portions of Lucide are held by Col... | Retain notice & license text |
| 33 | `object-assign` | 4.1.1 | Transitive | MIT | Copyright (c) Sindre Sorhus <sindresorhus@gmail.com>... | Retain notice & license text |
| 34 | `prop-types` | 15.8.1 | Transitive | MIT | Copyright (c) 2013-present, Facebook, Inc. | Retain notice & license text |
| 35 | `prop-types/react-is` | 16.13.1 | Transitive | MIT | Copyright (c) Facebook, Inc. and its affiliates. | Retain notice & license text |
| 36 | `react` | 18.3.1 | **Direct** | MIT | Copyright (c) Facebook, Inc. and its affiliates. | Retain notice & license text |
| 37 | `react-dom` | 18.3.1 | **Direct** | MIT | Copyright (c) Facebook, Inc. and its affiliates. | Retain notice & license text |
| 38 | `react-is` | 18.3.1 | Transitive | MIT | Copyright (c) Facebook, Inc. and its affiliates. | Retain notice & license text |
| 39 | `react-smooth` | 4.0.4 | Transitive | MIT | Copyright (c) 2016 recharts | Retain notice & license text |
| 40 | `react-transition-group` | 4.4.5 | Transitive | BSD-3-Clause | Copyright (c) 2018, React Community | Retain notice & license text |
| 41 | `recharts` | 2.15.4 | **Direct** | MIT | Copyright (c) 2015-present recharts | Retain notice & license text |
| 42 | `recharts-scale` | 0.4.5 | Transitive | MIT | Copyright (c) 2015 Sen Yang | Retain notice & license text |
| 43 | `scheduler` | 0.23.2 | Transitive | MIT | Copyright (c) Facebook, Inc. and its affiliates. | Retain notice & license text |
| 44 | `tailwind-merge` | 3.7.0 | **Direct** | MIT | Copyright (c) 2021 Dany Castillo | Retain notice & license text |
| 45 | `tiny-invariant` | 1.3.3 | Transitive | MIT | Copyright (c) 2019 Alexander Reardon | Retain notice & license text |
| 46 | `victory-vendor` | 36.9.2 | Transitive | MIT AND ISC | Copyright Formidable Labs, Inc. and D3 authors | Retain notice & license text |

### 2.3 Frontend Development & Build Tooling Environment (181 Packages)

The following **9 direct development packages** and **172 transitive tooling packages** (total: 181 packages) are utilized exclusively during compile, bundling, linting, and type-checking phases. They are not bundled into production runtime assets, but their licenses and copyrights are disclosed here for complete supply-chain transparency:

| Package Name | Exact Version | Scope | SPDX License Identifier | Primary Function / Tooling Role | Copyright Holder |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `@types/node` | 22.20.5 | **Direct** | MIT | TypeScript type definitions for Node.js runtime | DefinitelyTyped Contributors |
| `@types/react` | 18.3.31 | **Direct** | MIT | TypeScript definitions for React core | DefinitelyTyped Contributors |
| `@types/react-dom` | 18.3.7 | **Direct** | MIT | TypeScript definitions for React DOM | DefinitelyTyped Contributors |
| `@vitejs/plugin-react` | 4.7.0 | **Direct** | MIT | Vite plugin for React fast refresh and JSX | Vite Contributors |
| `autoprefixer` | 10.6.1 | **Direct** | MIT | PostCSS plugin to parse CSS and add vendor prefixes | Andrey Sitnik |
| `postcss` | 8.5.30 | **Direct** | MIT | CSS transformation and AST tooling framework | Andrey Sitnik |
| `tailwindcss` | 3.4.19 | **Direct** | MIT | Utility-first CSS generation framework | Tailwind Labs, Inc. |
| `typescript` | 5.9.3 | **Direct** | Apache-2.0 | Static type-checker and language compiler | Microsoft Corporation |
| `vite` | 6.4.4 | **Direct** | MIT | Next-generation frontend tooling and bundler (Rollup/esbuild) | Yuxi (Evan) You & Vite Contributors |

#### Transitive Development Tooling Summary (172 Packages)
- **License Distribution:** MIT (165 packages), ISC (10 packages), Apache-2.0 (4 packages: `typescript`, `baseline-browser-mapping`, `didyoumean`, `ts-interface-checker`), BSD-3-Clause (1 package: `source-map-js`), CC-BY-4.0 (1 package: `caniuse-lite`).
- **Special License Disclosures:**
  - `caniuse-lite` (v1.0.30001816): Licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0). Used solely as a read-only browser support database during CSS compilation.
  - `typescript` (v5.9.3): Licensed under Apache-2.0. Copyright (c) Microsoft Corporation. Used at build-time for type validation.

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
- **Notice:** This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. As permitted under MPL-2.0 Section 3.3 (Larger Works), NEXORA combines Certifi with other independent modules under the terms of the root MIT license, while preserving MPL-2.0 terms for Certifi itself.

### 3.5 FastAPI & Pydantic
- **FastAPI:** Copyright (c) 2018 Sebastián Ramírez (MIT)
- **Pydantic:** Copyright (c) 2017-present Pydantic Services Inc. (MIT)

### 3.6 Uvicorn & Starlette
- **Uvicorn:** Copyright (c) 2017-present Encode OSS Ltd (BSD-3-Clause)
- **Starlette:** Copyright (c) 2018 Encode OSS Ltd (BSD-3-Clause)

### 3.7 Packaging (packaging 26.3)
- **Dual License Election:** Disjunctive dual-licensed under Apache-2.0 OR BSD-2-Clause.
- **Election Disclosure:** Documented under the **BSD-2-Clause** license.
- **Copyright:** Copyright (c) Donald Stufft and individual contributors.

### 3.8 React & React-DOM (Bundled Web UI)
- **Packages:** `react` (v18.3.1), `react-dom` (v18.3.1), `scheduler` (v0.23.2)
- **License:** MIT License
- **Copyright:** Copyright (c) Meta Platforms, Inc. and its affiliates.
- **Distribution Note:** React and React-DOM are compiled and statically linked into `frontend/dist/assets/index-BhQcNE8G.js`. The production minifier retained 5 header comment blocks; this document provides the complete verbatim MIT license text.

### 3.9 Lucide React (Bundled Web UI)
- **Package:** `lucide-react` (v0.475.0)
- **License:** ISC License
- **Copyright:** Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
- **Distribution Note:** 56 distinct icon components are bundled into `frontend/dist/assets/index-BhQcNE8G.js`. Each bundled icon includes a 1-line ISC notice header in the bundle; the full ISC license text is provided in Section 4.7.

### 3.10 Recharts & Transitive D3 Visualization Subcomponents
- **Core Charting Engine:** `recharts` (v2.15.4, MIT, Copyright (c) 2015-present recharts).
- **D3 Visualization Subcomponents (Bundled):**
  - `d3-array` (v3.2.4, ISC, Copyright 2010-2023 Mike Bostock)
  - `d3-color` (v3.1.0, ISC, Copyright 2010-2022 Mike Bostock)
  - `d3-ease` (v3.0.1, BSD-3-Clause, Copyright 2010-2021 Mike Bostock)
  - `d3-format` (v3.1.2, ISC, Copyright 2010-2026 Mike Bostock)
  - `d3-interpolate` (v3.0.1, ISC, Copyright 2010-2021 Mike Bostock)
  - `d3-path` (v3.1.0, ISC, Copyright 2015-2022 Mike Bostock)
  - `d3-scale` (v4.0.2, ISC, Copyright 2010-2021 Mike Bostock)
  - `d3-shape` (v3.2.0, ISC, Copyright 2010-2022 Mike Bostock)
  - `d3-time` (v3.1.0, ISC, Copyright 2010-2022 Mike Bostock)
  - `d3-time-format` (v4.1.0, ISC, Copyright 2010-2021 Mike Bostock)
  - `d3-timer` (v3.0.1, ISC, Copyright 2010-2021 Mike Bostock)
  - `internmap` (v2.0.3, ISC, Copyright 2021 Mike Bostock)
- **Bundled Transitive Utilities:**
  - `lodash` (v4.18.1 / v4.17.21, MIT, Copyright OpenJS Foundation and other contributors)
  - `fast-equals` (v5.4.5, MIT, Copyright (c) 2025 Tony Quetano)
  - `react-smooth` (v4.0.4, MIT, Copyright (c) 2016 recharts)
  - `react-transition-group` (v4.4.5, BSD-3-Clause, Copyright (c) 2018, React Community)
  - `victory-vendor` (v36.9.2, MIT AND ISC, Copyright Formidable Labs, Inc.)
- **Notice Retention Legal Analysis:** Minification of the frontend bundle stripped comment blocks that lacked explicit `@license` tags. Consequently, in-bundle notices for Recharts, D3 modules, and Lodash are absent from `index-BhQcNE8G.js`. This document serves as the formal legal vehicle satisfying upstream notice retention and attribution obligations for all compiled web UI subcomponents.

### 3.11 Frontend Development Tooling & Compilers
- **TypeScript (v5.9.3):** Apache License 2.0. Copyright (c) Microsoft Corporation.
- **Vite (v6.4.4):** MIT License. Copyright (c) 2019-present Yuxi (Evan) You & Vite Contributors.
- **PostCSS (v8.5.30):** MIT License. Copyright (c) 2013 Andrey Sitnik.
- **Tailwind CSS (v3.4.19):** MIT License. Copyright (c) Tailwind Labs, Inc.

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
Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

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
Mozilla Public License Version 2.0
==================================

1. Definitions
--------------

1.1. "Contributor"
    means each individual or legal entity that creates, contributes to
    the creation of, or owns Covered Software.

1.2. "Contributor Version"
    means the combination of the Contributions of others (if any) used
    by a Contributor and that particular Contributor's Contribution.

1.3. "Contribution"
    means Covered Software of a particular Contributor.

1.4. "Covered Software"
    means Source Code Form to which the initial Contributor has attached
    the notice in Exhibit A, the Executable Form of such Source Code
    Form, and Modifications of such Source Code Form, in each case
    including portions thereof.

1.5. "Incompatible With Secondary Licenses"
    means

    (a) that the initial Contributor has attached the notice described
        in Exhibit B to the Covered Software; or

    (b) that the Covered Software was made available under the terms of
        version 1.1 or earlier of the License, but not also under the
        terms of a Secondary License.

1.6. "Executable Form"
    means any form of the work other than Source Code Form.

1.7. "Larger Work"
    means a work that combines Covered Software with other material, in
    a separate file or files, that is not Covered Software.

1.8. "License"
    means this document.

1.9. "Licensable"
    means having the right to grant, to the maximum extent possible,
    whether at the time of the initial grant or subsequently, any and
    all of the rights conveyed by this License.

1.10. "Modifications"
    means any of the following:

    (a) any file in Source Code Form that results from an addition to,
        deletion from, or modification of the contents of Covered
        Software; or

    (b) any new file in Source Code Form that contains any Covered
        Software.

1.11. "Patent Claims" of a Contributor
    means any patent claim(s), including without limitation, method,
    process, and apparatus claims, in any patent Licensable by such
    Contributor that would be infringed, but for the grant of the
    License, by the making, using, selling, offering for sale, having
    made, import, or transfer of either its Contributions or its
    Contributor Version.

1.12. "Secondary License"
    means either the GNU General Public License, Version 2.0, the GNU
    Lesser General Public License, Version 2.1, the GNU Affero General
    Public License, Version 3.0, or any later versions of those
    licenses.

1.13. "Source Code Form"
    means the form of the work preferred for making modifications.

1.14. "You" (or "Your")
    means an individual or a legal entity exercising rights under this
    License. For legal entities, "You" includes any entity that
    controls, is controlled by, or is under common control with You. For
    purposes of this definition, "control" means (a) the power, direct
    or indirect, to cause the direction or management of such entity,
    whether by contract or otherwise, or (b) ownership of more than
    fifty percent (50%) of the outstanding shares or beneficial
    ownership of such entity.

2. License Grants and Conditions
--------------------------------

2.1. Grants

Each Contributor hereby grants You a world-wide, royalty-free,
non-exclusive license:

(a) under intellectual property rights (other than patent or trademark)
    Licensable by such Contributor to use, reproduce, make available,
    modify, display, perform, distribute, and otherwise exploit its
    Contributions, either on an unmodified basis, with Modifications, or
    as part of a Larger Work; and

(b) under Patent Claims of such Contributor to make, use, sell, offer
    for sale, have made, import, and otherwise transfer either its
    Contributions or its Contributor Version.

2.2. Effective Date

The licenses granted in Section 2.1 with respect to any Contribution
become effective for each Contribution on the date the Contributor first
distributes such Contribution.

2.3. Limitations on Grant Scope

The licenses granted in this Section 2 are the only rights granted under
this License. No additional rights or licenses will be implied from the
distribution or licensing of Covered Software under this License.
Notwithstanding Section 2.1(b) above, no patent license is granted by a
Contributor:

(a) for any code that a Contributor has removed from Covered Software;
    or

(b) for infringements caused by: (i) Your and any other third party's
    modifications of Covered Software, or (ii) the combination of its
    Contributions with other software (except as part of its Contributor
    Version); or

(c) under Patent Claims infringed by Covered Software in the absence of
    its Contributions.

This License does not grant any rights in the trademarks, service marks,
or logos of any Contributor (except as may be necessary to comply with
the notice requirements in Section 3.4).

2.4. Subsequent Licenses

No Contributor makes additional grants as a result of Your choice to
distribute the Covered Software under a subsequent version of this
License (see Section 10.2) or under the terms of a Secondary License (if
permitted under the terms of Section 3.3).

2.5. Representation

Each Contributor represents that the Contributor believes its
Contributions are its original creation(s) or it has sufficient rights
to grant the rights to its Contributions conveyed by this License.

2.6. Fair Use

This License is not intended to limit any rights You have under
applicable copyright doctrines of fair use, fair dealing, or other
equivalents.

2.7. Conditions

Sections 3.1, 3.2, 3.3, and 3.4 are conditions of the licenses granted
in Section 2.1.

3. Responsibilities
-------------------

3.1. Distribution of Source Form

All distribution of Covered Software in Source Code Form, including any
Modifications that You create or to which You contribute, must be under
the terms of this License. You must inform recipients that the Source
Code Form of the Covered Software is governed by the terms of this
License, and how they can obtain a copy of this License. You may not
attempt to alter or restrict the recipients' rights in the Source Code
Form.

3.2. Distribution of Executable Form

If You distribute Covered Software in Executable Form then:

(a) such Covered Software must also be made available in Source Code
    Form, as described in Section 3.1, and You must inform recipients of
    the Executable Form how they can obtain a copy of such Source Code
    Form by reasonable means in a timely manner, at a charge no more
    than the cost of distribution to the recipient; and

(b) You may distribute such Executable Form under the terms of this
    License, or sublicense it under different terms, provided that the
    license for the Executable Form does not attempt to limit or alter
    the recipients' rights in the Source Code Form under this License.

3.3. Distribution of a Larger Work

You may create and distribute a Larger Work under terms of Your choice,
provided that You also comply with the requirements of this License for
the Covered Software. If the Larger Work is a combination of Covered
Software with a work governed by one or more Secondary Licenses, and the
Covered Software is not Incompatible With Secondary Licenses, this
License permits You to additionally distribute such Covered Software
under the terms of such Secondary License(s), so that the recipient of
the Larger Work may, at their option, further distribute the Covered
Software under the terms of either this License or such Secondary
License(s).

3.4. Notices

You may not remove or alter the substance of any license notices
(including copyright notices, patent notices, disclaimers of warranty,
or limitations of liability) contained within the Source Code Form of
the Covered Software, except that You may alter any license notices to
the extent required to remedy known factual inaccuracies.

3.5. Application of Additional Terms

You may choose to offer, and to charge a fee for, warranty, support,
indemnity or liability obligations to one or more recipients of Covered
Software. However, You may do so only on Your own behalf, and not on
behalf of any Contributor. You must make it absolutely clear that any
such warranty, support, indemnity, or liability obligation is offered by
You alone, and You hereby agree to indemnify every Contributor for any
liability incurred by such Contributor as a result of warranty, support,
indemnity or liability terms You offer. You may include additional
disclaimers of warranty and limitations of liability specific to any
jurisdiction.

4. Inability to Comply Due to Statute or Regulation
---------------------------------------------------

If it is impossible for You to comply with any of the terms of this
License with respect to some or all of the Covered Software due to
statute, judicial order, or regulation then You must: (a) comply with
the terms of this License to the maximum extent possible; and (b)
describe the limitations and the code they affect. Such description must
be placed in a text file included with all distributions of the Covered
Software under this License. Except to the extent prohibited by statute
or regulation, such description must be sufficiently detailed for a
recipient of ordinary skill to be able to understand it.

5. Termination
--------------

5.1. The rights granted under this License will terminate automatically
if You fail to comply with any of its terms. However, if You become
compliant, then the rights granted under this License from a particular
Contributor are reinstated (a) provisionally, unless and until such
Contributor explicitly and finally terminates Your grants, and (b) on an
ongoing basis, if such Contributor fails to notify You of the
non-compliance by some reasonable means prior to 60 days after You have
come back into compliance. Moreover, Your grants from a particular
Contributor are reinstated on an ongoing basis if such Contributor
notifies You of the non-compliance by some reasonable means, this is the
first time You have received notice of non-compliance with this License
from such Contributor, and You become compliant prior to 30 days after
Your receipt of the notice.

5.2. If You initiate litigation against any entity by asserting a patent
infringement claim (excluding declaratory judgment actions,
counter-claims, and cross-claims) alleging that a Contributor Version
directly or indirectly infringes any patent, then the rights granted to
You by any and all Contributors for the Covered Software under Section
2.1 of this License shall terminate.

5.3. In the event of termination under Sections 5.1 or 5.2 above, all
end user license agreements (excluding distributors and resellers) which
have been validly granted by You or Your distributors under this License
prior to termination shall survive termination.

************************************************************************
*                                                                      *
*  6. Disclaimer of Warranty                                           *
*  -------------------------                                           *
*                                                                      *
*  Covered Software is provided under this License on an "as is"       *
*  basis, without warranty of any kind, either expressed, implied, or  *
*  statutory, including, without limitation, warranties that the       *
*  Covered Software is free of defects, merchantable, fit for a        *
*  particular purpose or non-infringing. The entire risk as to the     *
*  quality and performance of the Covered Software is with You.        *
*  Should any Covered Software prove defective in any respect, You     *
*  (not any Contributor) assume the cost of any necessary servicing,   *
*  repair, or correction. This disclaimer of warranty constitutes an   *
*  essential part of this License. No use of any Covered Software is   *
*  authorized under this License except under this disclaimer.         *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  7. Limitation of Liability                                          *
*  --------------------------                                          *
*                                                                      *
*  Under no circumstances and under no legal theory, whether tort      *
*  (including negligence), contract, or otherwise, shall any           *
*  Contributor, or anyone who distributes Covered Software as          *
*  permitted above, be liable to You for any direct, indirect,         *
*  special, incidental, or consequential damages of any character      *
*  including, without limitation, damages for lost profits, loss of    *
*  goodwill, work stoppage, computer failure or malfunction, or any    *
*  and all other commercial damages or losses, even if such party      *
*  shall have been informed of the possibility of such damages. This   *
*  limitation of liability shall not apply to liability for death or   *
*  personal injury resulting from such party's negligence to the       *
*  extent applicable law prohibits such limitation. Some               *
*  jurisdictions do not allow the exclusion or limitation of           *
*  incidental or consequential damages, so this exclusion and          *
*  limitation may not apply to You.                                    *
*                                                                      *
************************************************************************

8. Litigation
-------------

Any litigation relating to this License may be brought only in the
courts of a jurisdiction where the defendant maintains its principal
place of business and such litigation shall be governed by laws of that
jurisdiction, without reference to its conflict-of-law provisions.
Nothing in this Section shall prevent a party's ability to bring
cross-claims or counter-claims.

9. Miscellaneous
----------------

This License represents the complete agreement concerning the subject
matter hereof. If any provision of this License is held to be
unenforceable, such provision shall be reformed only to the extent
necessary to make it enforceable. Any law or regulation which provides
that the language of a contract shall be construed against the drafter
shall not be used to construe this License against a Contributor.

10. Versions of the License
---------------------------

10.1. New Versions

Mozilla Foundation is the license steward. Except as provided in Section
10.3, no one other than the license steward has the right to modify or
publish new versions of this License. Each version will be given a
distinguishing version number.

10.2. Effect of New Versions

You may distribute the Covered Software under the terms of the version
of the License under which You originally received the Covered Software,
or under the terms of any subsequent version published by the license
steward.

10.3. Modified Versions

If you create software not governed by this License, and you want to
create a new license for such software, you may create and use a
modified version of this License if you rename the license and remove
any references to the name of the license steward (except to note that
such modified license differs from this License).

10.4. Distributing Source Code Form that is Incompatible With Secondary
Licenses

If You choose to distribute Source Code Form that is Incompatible With
Secondary Licenses under the terms of this version of the License, the
notice described in Exhibit B of this License must be attached.

Exhibit A - Source Code Form License Notice
-------------------------------------------

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at https://mozilla.org/MPL/2.0/.

If it is not possible or desirable to put the notice in a particular
file, then You may include the notice in a location (such as a LICENSE
file in a relevant directory) where a recipient would be likely to look
for such a notice.

You may add additional accurate notices of copyright ownership.

Exhibit B - "Incompatible With Secondary Licenses" Notice
---------------------------------------------------------

  This Source Code Form is "Incompatible With Secondary Licenses", as
  defined by the Mozilla Public License, v. 2.0.
```

### 4.6 Python Software Foundation License Version 2.0 (PSF-2.0)
```text
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation;
All Rights Reserved" are retained in Python alone or in any derivative version
prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
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
3. **Downstream Dual-License Election:**
   - Disjunctive dual-licensed dependencies (e.g. `packaging` under BSD-2-Clause / Apache-2.0; `victory-vendor` under MIT AND ISC) are disclosed under their respective governing terms. Final legal ratification of dual-license elections remains reserved for the Project Owner.

