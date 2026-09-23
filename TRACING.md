# Optional OpenTelemetry and Phoenix tracing

Tracing is off by default. The runtime still records its local result, grade, calls, and events when tracing is off; enabling tracing builds a separate, payload-free span tree after execution. The tree has one experiment span, one span per case/repetition, one system span per attempted arm, one span per agent execution, and one span per model/tool call. Handoff, gate, and recovery decisions are span events with correlation IDs. A cancelled or failed attempt still receives closed spans. A failed exporter returns a visible warning and cannot change predictions, grades, or accounting.

For a local in-memory trace, set `telemetry.enabled: true` and `telemetry.exporter: memory` in the experiment YAML, or pass `--trace` to `run` or `compare`. `--no-trace` overrides an enabled file. The CLI prints the span count and warnings to stderr; stdout remains result JSON. The `trace_experiment` Python API returns the `SpanRecord` tuple for programmatic inspection or a custom `SpanSink`.

To export to an OTLP HTTP traces endpoint, install the optional `otel` extra if you want this adapter, then configure environment references:

```yaml
telemetry:
  enabled: true
  exporter: otlp_http
  endpoint_env: PHOENIX_OTLP_TRACES_ENDPOINT
  header_env:
    authorization: PHOENIX_AUTHORIZATION
```

Set `PHOENIX_OTLP_TRACES_ENDPOINT` to the actual `/v1/traces` HTTP endpoint and `PHOENIX_AUTHORIZATION` to the header value required by your Phoenix deployment. The YAML and safe configuration snapshot contain only variable names, never their values. [Phoenix's endpoint guide](https://arize.com/docs/phoenix/learn/faqs/what-is-my-phoenix-endpoint) distinguishes the application URL from the trace endpoint, and its [authentication guide](https://arize.com/docs/phoenix/deployment/authentication) describes header options. The bridge uses the [OpenTelemetry Python OTLP HTTP exporter](https://opentelemetry.io/docs/languages/python/exporters/) without global auto-instrumentation. A connected Phoenix smoke test is opt-in and has not been run in the offline fixture environment.

The default span attributes omit case text, prompts, tool arguments, outputs, API keys, and header values. The `header_env` names are validated before inference when export is enabled; missing optional SDK or exporter failure becomes a warning. No collector, Phoenix process, or OTel SDK is needed for the default clone-based run.
