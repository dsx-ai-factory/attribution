import json
from datetime import datetime, timezone

from attribution_blueprint.connectors.logs.service_observation_adapter import adapt_log_observation_for_service
from attribution_blueprint.types import Format, Source


def test_raw_syslog_adapted_to_service_loki_dict_list():
    observation = {
        "context": {
            "source": Source.SYSLOG,
            "format": Format.RAW_LOG,
            "resources": "node-a",
            "time_interval": {
                "start": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
            },
        },
        "body": "2026-05-01T00:00:03Z node-a kernel\nnode-a nvrm\n",
    }

    adapted = adapt_log_observation_for_service(observation)

    assert observation["context"]["format"] == Format.RAW_LOG
    assert adapted["context"]["format"] == Format.LOKI_DICT_LIST
    assert adapted["context"]["resources"] == "AllJobResources"
    assert json.loads(adapted["body"]) == [
        {
            "timestamp": 1777593603000000000,
            "body": "2026-05-01T00:00:03Z node-a kernel",
            "node": "node-a",
        },
        {
            "timestamp": 1777593600000000000,
            "body": "node-a nvrm",
            "node": "node-a",
        },
    ]


def test_raw_syslog_adapter_can_infer_node_from_line():
    observation = {
        "context": {
            "source": "syslog",
            "format": "raw_log",
            "resources": "AllJobResources",
            "time_interval": {
                "start": "2026-05-01T00:00:00+00:00",
                "end": "2026-05-01T01:00:00+00:00",
            },
        },
        "body": "2026-05-01T00:00:03Z node-b kernel\n",
    }

    adapted = adapt_log_observation_for_service(observation)

    assert json.loads(adapted["body"])[0]["node"] == "node-b"


def test_non_raw_syslog_observation_is_unchanged():
    observation = {
        "context": {
            "source": Source.APPLOG,
            "format": Format.RAW_LOG,
            "resources": "AllJobResources",
            "time_interval": {
                "start": datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 5, 1, 1, 0, 0, tzinfo=timezone.utc),
            },
        },
        "body": "rank log\n",
    }

    assert adapt_log_observation_for_service(observation) is observation
