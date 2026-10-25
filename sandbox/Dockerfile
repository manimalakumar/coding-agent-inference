# Trusted setup only. Exact base/uv digests are resolved and recorded by prepare.py.
ARG BASE
ARG UV_IMAGE
FROM ${UV_IMAGE} AS uv
FROM ${BASE} AS build
COPY --from=uv /uv /usr/local/bin/uv
COPY source /workspace
COPY requirements.in /opt/requirements.in
WORKDIR /workspace/libs/partners/openai
# Do not substitute newer dependencies when historical pins are incompatible.
RUN uv venv --python /usr/local/bin/python /opt/venv \
 && uv pip compile /opt/requirements.in --python /opt/venv/bin/python \
      --generate-hashes -o /opt/requirements.lock \
 && uv pip sync --python /opt/venv/bin/python /opt/requirements.lock
RUN /opt/venv/bin/python -c 'import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions())},indent=2))' > /opt/environment.json
# Prefetch tokenization assets during setup, never during agent execution.
RUN TIKTOKEN_CACHE_DIR=/opt/tiktoken /opt/venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base"); tiktoken.get_encoding("o200k_base")'
FROM ${BASE}
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/tiktoken /opt/tiktoken
COPY --from=build /opt/requirements.lock /opt/requirements.lock
COPY --from=build /opt/environment.json /opt/environment.json
USER 65532:65532
WORKDIR /tmp
CMD ["/opt/venv/bin/python", "--version"]
