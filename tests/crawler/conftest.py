"""Expose the release suite's strict real-prefix fixtures to acceptance tests."""

from menagerie.crawler.tests.conftest import (  # noqa: F401
    real_environment_fixture,
    real_environment_seal_counter,
)
