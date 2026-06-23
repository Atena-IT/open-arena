{{/*
Expand the name of the chart.
*/}}
{{- define "open-arena.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
Truncate at 63 chars because some Kubernetes name fields are limited.
If release name contains chart name it will be used as a full name.
*/}}
{{- define "open-arena.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label value.
*/}}
{{- define "open-arena.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "open-arena.labels" -}}
helm.sh/chart: {{ include "open-arena.chart" . }}
{{ include "open-arena.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "open-arena.selectorLabels" -}}
app.kubernetes.io/name: {{ include "open-arena.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "open-arena.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "open-arena.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Compute the ingress hostname.
When .Values.ingress.useOrgHostPattern is true and .Values.org is set,
renders: arena.<org>.dev.reply-modelfactory.com
Otherwise falls back to the first host in .Values.ingress.hosts.
*/}}
{{- define "open-arena.ingressHost" -}}
{{- if and .Values.ingress.useOrgHostPattern .Values.org }}
{{- printf "arena.%s.dev.reply-modelfactory.com" .Values.org }}
{{- else if .Values.ingress.hosts }}
{{- (index .Values.ingress.hosts 0).host }}
{{- else }}
{{- "arena.local" }}
{{- end }}
{{- end }}

{{/*
Name of the ConfigMap holding non-secret env vars.
*/}}
{{- define "open-arena.configMapName" -}}
{{- printf "%s-config" (include "open-arena.fullname" .) }}
{{- end }}

{{/*
Name of the Secret holding sensitive env vars.
Respects existingSecret override.
*/}}
{{- define "open-arena.secretName" -}}
{{- if .Values.existingSecret }}
{{- .Values.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "open-arena.fullname" .) }}
{{- end }}
{{- end }}
