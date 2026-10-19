# Third-party materials and license exceptions

NVIDIA Attribution project code is licensed under [Apache-2.0](LICENSE).
The materials below retain their respective notices and terms.

## shadcn/ui

The WebUI components in
[the frontend UI component directory](blueprint/src/attribution_blueprint/webui/frontend/src/components/ui),
together with supporting hooks, utilities, and styles, include code derived
from [shadcn/ui](https://github.com/shadcn-ui/ui) and modified for NVIDIA
Attribution. The upstream code is licensed under MIT
([original license](https://github.com/shadcn-ui/ui/blob/6ea6856f5a1082d4d9c231559b6bc3ee73827493/LICENSE.md)):

```text
MIT License

Copyright (c) 2023 shadcn

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

## OpenAPI Generator templates

The templates in [service/service-app/openapi/templates](service/service-app/openapi/templates)
are derived from [OpenAPI Generator](https://github.com/OpenAPITools/openapi-generator).
Their [upstream Apache-2.0 license and copyright notices](service/service-app/openapi/templates/LICENSE)
are preserved. Local changes provide the project header and adapt generated
client arguments. Generated client code uses the project's Apache-2.0 license.

## Contributor Covenant

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) is adapted from Contributor Covenant
3.0 and retains its Creative Commons Attribution-ShareAlike 4.0 license and
attribution.

## Installed dependencies

Python and JavaScript dependencies retain their own licenses. Package manifests
and lockfiles record the dependencies used by the Service, Library, Blueprint,
and interactive documentation. Review the license files and notices distributed
with those dependencies when redistributing a built application or documentation
site. The project's Apache-2.0 license does not replace those terms. See
[Python dependencies](#python-dependencies) for the direct Python dependencies
and their licenses.

The Blueprint WebUI and interactive walkthrough builds write
`assets/THIRD_PARTY_NOTICES.txt`. It reproduces this document and the license
and NOTICE texts of the JavaScript packages bundled into that build. Keep that
file with the built assets when redistributing them.

## Python dependencies

The tables in this section list the direct third-party Python dependencies of
each component. Each version is the version in the component lockfile. We
checked each license against the package metadata and license files for that
version on PyPI.

This repository does not contain a copy of these packages. A package manager
downloads each package from PyPI. Each package distribution contains its own
license files and notices. The "Distribution" column uses these values:

- **Installed separately**: A package manager installs the package with the
  component. A container image that you build from this repository also
  contains the package.
- **Installed separately (`name` extra)**: A package manager installs the
  package only when you select that optional extra.
- **Build/test only**: The project uses the package only to build, generate,
  lint, or test code. The installed component does not need the package.

The lockfiles also record transitive dependencies, which these tables do not
list. Each component builds with hatchling (MIT), and each `pyproject.toml`
file requires `hatchling>=1.27`. The Library and Blueprint lockfiles do not
record a hatchling version.

### Library (`attribution-lib`)

Versions come from `service/attribution-lib/uv.lock`.

| Package | Version | License | Source and license reference | Distribution |
| --- | --- | --- | --- | --- |
| duckdb | 1.4.3 | MIT | [Source](https://github.com/duckdb/duckdb-python), [PyPI release](https://pypi.org/project/duckdb/1.4.3/) | Installed separately |
| loguru | 0.7.3 | MIT | [Source](https://github.com/Delgan/loguru), [PyPI release](https://pypi.org/project/loguru/0.7.3/) | Installed separately |
| packaging | 26.1 | Apache-2.0 OR BSD-2-Clause | [Source](https://github.com/pypa/packaging), [PyPI release](https://pypi.org/project/packaging/26.1/) | Installed separately |
| pandas | 3.0.2 | BSD-3-Clause | [Source](https://github.com/pandas-dev/pandas), [PyPI release](https://pypi.org/project/pandas/3.0.2/) | Installed separately |
| pyahocorasick | 2.3.0 | BSD-3-Clause; some files are public domain | [Source](https://github.com/WojciechMula/pyahocorasick), [PyPI release](https://pypi.org/project/pyahocorasick/2.3.0/) | Installed separately |
| pydantic | 2.13.3 | MIT | [Source](https://github.com/pydantic/pydantic), [PyPI release](https://pypi.org/project/pydantic/2.13.3/) | Installed separately |
| pytz | 2026.1.post1 | MIT | [Source](https://github.com/stub42/pytz), [PyPI release](https://pypi.org/project/pytz/2026.1.post1/) | Installed separately |
| simpleeval | 1.0.7 | MIT | [Source](https://github.com/danthedeckie/simpleeval), [PyPI release](https://pypi.org/project/simpleeval/1.0.7/) | Installed separately |
| tzlocal | 5.3.1 | MIT | [Source](https://github.com/regebro/tzlocal), [PyPI release](https://pypi.org/project/tzlocal/5.3.1/) | Installed separately |
| mypy | 1.20.2 | MIT | [Source](https://github.com/python/mypy), [PyPI release](https://pypi.org/project/mypy/1.20.2/) | Build/test only |
| openpyxl | 3.1.5 | MIT | [Source](https://foss.heptapod.net/openpyxl/openpyxl), [PyPI release](https://pypi.org/project/openpyxl/3.1.5/) | Build/test only |
| pre-commit | 4.6.0 | MIT | [Source](https://github.com/pre-commit/pre-commit), [PyPI release](https://pypi.org/project/pre-commit/4.6.0/) | Build/test only |
| pytest | 9.0.3 | MIT | [Source](https://github.com/pytest-dev/pytest), [PyPI release](https://pypi.org/project/pytest/9.0.3/) | Build/test only |
| pytest-benchmark | 5.2.3 | BSD-2-Clause | [Source](https://github.com/ionelmc/pytest-benchmark), [PyPI release](https://pypi.org/project/pytest-benchmark/5.2.3/) | Build/test only |
| pytest-cov | 7.1.0 | MIT | [Source](https://github.com/pytest-dev/pytest-cov), [PyPI release](https://pypi.org/project/pytest-cov/7.1.0/) | Build/test only |
| pytest-loguru | 0.4.0 | MIT | [Source](https://github.com/mcarans/pytest-loguru), [PyPI release](https://pypi.org/project/pytest-loguru/0.4.0/) | Build/test only |
| pytest-xdist | 3.8.0 | MIT | [Source](https://github.com/pytest-dev/pytest-xdist), [PyPI release](https://pypi.org/project/pytest-xdist/3.8.0/) | Build/test only |
| ruff | 0.15.11 | MIT | [Source](https://github.com/astral-sh/ruff), [PyPI release](https://pypi.org/project/ruff/0.15.11/) | Build/test only |
| ty | 0.0.32 | MIT | [Source](https://github.com/astral-sh/ty), [PyPI release](https://pypi.org/project/ty/0.0.32/) | Build/test only |
| zstd | 1.5.7.3 | BSD-2-Clause | [Source](https://github.com/sergey-dryabzhinsky/python-zstd), [PyPI release](https://pypi.org/project/zstd/1.5.7.3/) | Build/test only |

### Service (`nvidia-attribution-service`)

Versions come from `service/service-app/uv.lock`. The
`openapi-generator-cli[jdk4py]` development dependency installs jdk4py, which
supplies the Java runtime that runs OpenAPI Generator.

| Package | Version | License | Source and license reference | Distribution |
| --- | --- | --- | --- | --- |
| fastapi | 0.135.3 | MIT | [Source](https://github.com/fastapi/fastapi), [PyPI release](https://pypi.org/project/fastapi/0.135.3/) | Installed separately |
| h11 | 0.16.0 | MIT | [Source](https://github.com/python-hyper/h11), [PyPI release](https://pypi.org/project/h11/0.16.0/) | Installed separately |
| loguru | 0.7.3 | MIT | [Source](https://github.com/Delgan/loguru), [PyPI release](https://pypi.org/project/loguru/0.7.3/) | Installed separately |
| openpyxl | 3.1.5 | MIT | [Source](https://foss.heptapod.net/openpyxl/openpyxl), [PyPI release](https://pypi.org/project/openpyxl/3.1.5/) | Installed separately |
| opentelemetry-exporter-otlp-proto-grpc | 1.43.0 | Apache-2.0 | [Source](https://github.com/open-telemetry/opentelemetry-python), [PyPI release](https://pypi.org/project/opentelemetry-exporter-otlp-proto-grpc/1.43.0/) | Installed separately |
| opentelemetry-sdk | 1.43.0 | Apache-2.0 | [Source](https://github.com/open-telemetry/opentelemetry-python), [PyPI release](https://pypi.org/project/opentelemetry-sdk/1.43.0/) | Installed separately |
| prometheus-client | 0.25.0 | Apache-2.0 AND BSD-2-Clause | [Source](https://github.com/prometheus/client_python), [PyPI release](https://pypi.org/project/prometheus-client/0.25.0/) | Installed separately |
| prometheus-fastapi-instrumentator | 8.0.2 | ISC | [Source](https://github.com/trallnag/prometheus-fastapi-instrumentator), [PyPI release](https://pypi.org/project/prometheus-fastapi-instrumentator/8.0.2/) | Installed separately |
| pydantic | 2.13.0 | MIT | [Source](https://github.com/pydantic/pydantic), [PyPI release](https://pypi.org/project/pydantic/2.13.0/) | Installed separately |
| uvicorn | 0.44.0 | BSD-3-Clause | [Source](https://github.com/Kludex/uvicorn), [PyPI release](https://pypi.org/project/uvicorn/0.44.0/) | Installed separately |
| hatchling | 1.29.0 | MIT | [Source](https://github.com/pypa/hatch), [PyPI release](https://pypi.org/project/hatchling/1.29.0/) | Build/test only |
| jdk4py | 21.0.8.2 | GPL-2.0-only | [Source](https://github.com/activeviam/jdk4py), [PyPI release](https://pypi.org/project/jdk4py/21.0.8.2/) | Build/test only |
| mypy | 1.20.1 | MIT | [Source](https://github.com/python/mypy), [PyPI release](https://pypi.org/project/mypy/1.20.1/) | Build/test only |
| openapi-generator-cli | 7.21.0 | Apache-2.0 | [Source](https://github.com/OpenAPITools/openapi-generator), [PyPI release](https://pypi.org/project/openapi-generator-cli/7.21.0/) | Build/test only |
| psutil | 7.2.2 | BSD-3-Clause | [Source](https://github.com/giampaolo/psutil), [PyPI release](https://pypi.org/project/psutil/7.2.2/) | Build/test only |
| pytest-cov | 7.1.0 | MIT | [Source](https://github.com/pytest-dev/pytest-cov), [PyPI release](https://pypi.org/project/pytest-cov/7.1.0/) | Build/test only |
| pyzstd | 0.19.1 | BSD-3-Clause | [Source](https://github.com/Rogdham/pyzstd), [PyPI release](https://pypi.org/project/pyzstd/0.19.1/) | Build/test only |
| ruff | 0.15.10 | MIT | [Source](https://github.com/astral-sh/ruff), [PyPI release](https://pypi.org/project/ruff/0.15.10/) | Build/test only |
| schemathesis | 4.15.2 | MIT | [Source](https://github.com/schemathesis/schemathesis), [PyPI release](https://pypi.org/project/schemathesis/4.15.2/) | Build/test only |

### Generated client (`nvidia-attribution-client`)

The generated client does not have its own lockfile. Versions come from
`service/service-app/uv.lock`, which installs the client for the Service tests.

| Package | Version | License | Source and license reference | Distribution |
| --- | --- | --- | --- | --- |
| pydantic | 2.13.0 | MIT | [Source](https://github.com/pydantic/pydantic), [PyPI release](https://pypi.org/project/pydantic/2.13.0/) | Installed separately |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause | [Source](https://github.com/dateutil/dateutil), [PyPI release](https://pypi.org/project/python-dateutil/2.9.0.post0/) | Installed separately |
| typing-extensions | 4.16.0 | PSF-2.0 | [Source](https://github.com/python/typing_extensions), [PyPI release](https://pypi.org/project/typing-extensions/4.16.0/) | Installed separately |
| urllib3 | 2.8.0 | MIT | [Source](https://github.com/urllib3/urllib3), [PyPI release](https://pypi.org/project/urllib3/2.8.0/) | Installed separately |
| hatchling | 1.29.0 | MIT | [Source](https://github.com/pypa/hatch), [PyPI release](https://pypi.org/project/hatchling/1.29.0/) | Build/test only |

### Blueprint (`nvidia-attribution-blueprint`)

Versions come from `blueprint/uv.lock`.

| Package | Version | License | Source and license reference | Distribution |
| --- | --- | --- | --- | --- |
| loguru | 0.7.3 | MIT | [Source](https://github.com/Delgan/loguru), [PyPI release](https://pypi.org/project/loguru/0.7.3/) | Installed separately |
| requests | 2.33.1 | Apache-2.0 | [Source](https://github.com/psf/requests), [PyPI release](https://pypi.org/project/requests/2.33.1/) | Installed separately |
| fastapi | 0.139.0 | MIT | [Source](https://github.com/fastapi/fastapi), [PyPI release](https://pypi.org/project/fastapi/0.139.0/) | Installed separately (`web` extra) |
| kubernetes | 36.0.3 | Apache-2.0 | [Source](https://github.com/kubernetes-client/python), [PyPI release](https://pypi.org/project/kubernetes/36.0.3/) | Installed separately (`pod-placement-recorder` extra) |
| PyYAML | 6.0.3 | MIT | [Source](https://github.com/yaml/pyyaml), [PyPI release](https://pypi.org/project/PyYAML/6.0.3/) | Installed separately (`web` extra) |
| uvicorn | 0.46.0 | BSD-3-Clause | [Source](https://github.com/Kludex/uvicorn), [PyPI release](https://pypi.org/project/uvicorn/0.46.0/) | Installed separately (`web` extra) |
| httpx | 0.28.1 | BSD-3-Clause | [Source](https://github.com/encode/httpx), [PyPI release](https://pypi.org/project/httpx/0.28.1/) | Build/test only |
| pytest | 9.0.3 | MIT | [Source](https://github.com/pytest-dev/pytest), [PyPI release](https://pypi.org/project/pytest/9.0.3/) | Build/test only |
| pytest-cov | 7.1.0 | MIT | [Source](https://github.com/pytest-dev/pytest-cov), [PyPI release](https://pypi.org/project/pytest-cov/7.1.0/) | Build/test only |
| pytest-mock | 3.15.1 | MIT | [Source](https://github.com/pytest-dev/pytest-mock), [PyPI release](https://pypi.org/project/pytest-mock/3.15.1/) | Build/test only |
| ruff | 0.15.12 | MIT | [Source](https://github.com/astral-sh/ruff), [PyPI release](https://pypi.org/project/ruff/0.15.12/) | Build/test only |
