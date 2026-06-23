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

## ModelFactory org-node integration

The chart is designed to be embedded as a sub-chart inside the ModelFactory
**org-node** Helm chart. Each organization gets its own release with
per-org values injected by the parent chart.

In the org-node `Chart.yaml`:

```yaml
dependencies:
  - name: open-arena
    version: "0.1.0"
    repository: "oci://ghcr.io/atenareply/helm"
```

In the org-node `values.yaml` (per-org override):

```yaml
open-arena:
  org: acme                         # sets ingress host: arena.acme.dev.reply-modelfactory.com
  image:
    tag: "1.2.3"
  ingress:
    enabled: true
    className: nginx
  secrets:
    OPEN_ARENA_API_TOKEN: "..."
    GITEA_TOKEN: "..."
  env:
    GITEA_BASE_URL: "https://gitea.acme.internal"
    GITEA_ORG: "acme"
```

The chart computes the ingress hostname as
`arena.<org>.dev.reply-modelfactory.com` when `ingress.useOrgHostPattern`
is `true` (the default) and `org` is non-empty.

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
| `ingress` | Enable / configure the Ingress resource |
| `autoscaling` | HPA (disabled by default) |

## Environment variable contract

See [`../../deploy/README.md`](../../deploy/README.md) for the full env var
contract and local-dev instructions.
