# NVIDIA Attribution Client

Python client for the NVIDIA Attribution Service HTTP API.

> **WARN: Generated code.** This package is produced by
> [openapi-generator](https://openapi-generator.tech) from the service's own
> OpenAPI schema. Do not edit by hand -- change the service routes and
> regenerate. See `service/service-app/openapi/README.md`.

## Usage

```python
import nvidia_attribution_client
from nvidia_attribution_client.rest import ApiException

configuration = nvidia_attribution_client.Configuration(host="http://localhost:8000")

with nvidia_attribution_client.ApiClient(configuration) as api_client:
    api = nvidia_attribution_client.DefaultApi(api_client)

    # Register a workload, returning a handle.
    att_id = api.new_attribution_latest_attributor_post(info).attributor_id

    # Feed it observations (logs, metrics).
    api.ingest_latest_attributor_att_id_observation_post(att_id, observation)

    # Then ask for an attribution.
    attribution = api.attribute_latest_attributor_att_id_attribution_get(att_id)

    # Finally, clean up the handle.
    api.delete_attribution_latest_attributor_att_id_delete(att_id)
```

Every optional parameter is keyword-only, so adding one to the API in a later
release cannot silently rebind an existing positional argument.

Errors surface as `ApiException`, which carries `.status` and `.body`.
