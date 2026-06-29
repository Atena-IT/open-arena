# open-arena Helm chart

Deploys the [Open Arena](https://github.com/atenareply/open-arena) FastAPI
service on Kubernetes.

## Standalone deployment

```bash
# Add your values
cp helm/open-arena/values.yaml my-values.yaml
# Edit my-values.yaml — at minimum set secrets.OPEN_ARENA_API_TOKEN

helm upgrade --install open-arena helm/open-arena \
  --namespace open-arena --create-namespace \
  --values my-values.yaml
```

## Embedding as a sub-chart

The chart is designed to be embedded as a sub-chart inside a parent
(umbrella) Helm chart. Each organization gets its own release with
per-org values injected by the parent chart.

In the parent chart's `Chart.yaml`:

```yaml
dependencies:
  - name: open-arena
    version: "0.1.0"
    repository: "oci://ghcr.io/atenareply/helm"
```

In the parent chart's `values.yaml` (per-org override):

```yaml
open-arena:
  org: acme                         # sets ingress host: arena.acme.<baseDomain>
  image:
    tag: "1.2.3"
  ingress:
    enabled: true
    className: nginx
    baseDomain: example.com         # -> arena.acme.example.com
  secrets:
    OPEN_ARENA_API_TOKEN: "..."
    GITEA_TOKEN: "..."
  env:
    GITEA_BASE_URL: "https://gitea.acme.internal"
    GITEA_ORG: "acme"
```

The chart computes the ingress hostname as
`arena.<org>.<ingress.baseDomain>` when `ingress.useOrgHostPattern`
is `true` (the default) and `org` is non-empty (`baseDomain` defaults to
`example.com` — set it to your cluster's DNS domain).

## Secrets & API token

`OPEN_ARENA_API_TOKEN` gates every `/v1/*` request. **If it is left empty and no
`existingSecret` is set, the API still starts but falls back to the built-in
development token `open-arena-dev-token`, which is public knowledge** — never run
that way outside local development.

Three ways to configure it, in increasing order of production-readiness:

1. **Inline** (dev/test only): set `secrets.OPEN_ARENA_API_TOKEN` in your values
   file. Helm base64-encodes it into a managed Secret (still plain text in the
   release).
2. **Externally-managed Secret**: set `existingSecret: <name>` and let the chart
   read the token from a Secret you provision out-of-band. The `secrets:` block is
   then ignored.
3. **External Secrets Operator (ESO)** — recommended. Provision the Secret from
   your secrets backend (Vault, AWS/GCP Secrets Manager, …) with an `ExternalSecret`,
   then point the chart at it via `existingSecret`:

   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: ExternalSecret
   metadata:
     name: open-arena-secrets
   spec:
     refreshInterval: 1h
     secretStoreRef:
       name: my-secret-store
       kind: ClusterSecretStore
     target:
       name: open-arena-secrets        # -> set existingSecret: open-arena-secrets
     data:
       - secretKey: OPEN_ARENA_API_TOKEN
         remoteRef:
           key: open-arena/api-token
   ```

To make a missing token a hard error instead of a warning, set
`security.requireApiToken: true` — `helm install/upgrade` (and `helm template`)
then fails unless a token is provided via `secrets.OPEN_ARENA_API_TOKEN` or
`existingSecret`.

## Values reference

See [`values.yaml`](values.yaml) for the full annotated reference. Key
sections:

| Section | Purpose |
|---|---|
| `org` | Organization slug; drives ingress host templating |
| `image` | Repository, tag, pull policy |
| `env` | Non-secret env vars → ConfigMap |
| `secrets` | Sensitive env vars → Secret (or `existingSecret`) |
| `persistence` | `.open-arena/` PVC for trial cache / SQLite |
| `ingress` | Enable / configure the Ingress resource (incl. `baseDomain` for the per-org host pattern) |
| `autoscaling` | HPA (disabled by default) |
| `security` | `requireApiToken` — hard-fail installs that configure no API token |

## Environment variable contract

See [`../../deploy/README.md`](../../deploy/README.md) for the full env var
contract and local-dev instructions.
