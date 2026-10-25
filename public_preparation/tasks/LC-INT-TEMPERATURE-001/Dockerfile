FROM ghcr.io/astral-sh/uv@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv
FROM python@sha256:4c92ffcde4dd6f1ff72a24518f49fd4990b27134987dfa31a733badde66df9f8
COPY --from=uv /uv /usr/local/bin/uv
COPY source /workspace
COPY expected-requirements.lock /tmp/expected-requirements.lock
WORKDIR /workspace/libs/core
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
RUN uv sync --frozen --group test --group lint --group typing --group test_integration
RUN uv pip freeze --python /opt/venv/bin/python > /tmp/actual-requirements.lock && diff -u /tmp/expected-requirements.lock /tmp/actual-requirements.lock
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/libs/core:/workspace/libs/standard-tests
ENV PYTEST_ADDOPTS="-p no:cacheprovider" MYPY_CACHE_DIR=/dev/null
USER 65532
