# Local Kubernetes (kind or minikube)

Advisory API, Streamlit dashboard, and an MLflow tracking server. Nothing here
actuates plant equipment. Do not commit real database passwords; `secret.example.yaml`
uses the placeholder `CHANGE_ME`.

The MLflow `PersistentVolumeClaim` (`cement-copilot-mlruns`) is for a **local kind
or minikube cluster only**. It stores a SQLite tracking backend and the `mlruns`
artifact directory on the node disk. It is not a production model store. Use a
managed database and object storage for anything beyond a laptop cluster.

## Build images

```bash
docker build -t cement-kiln-copilot-api:local .
docker build -f Dockerfile.streamlit -t cement-kiln-copilot-dashboard:local .
```

### kind

```bash
kind create cluster --name cement-copilot
kind load docker-image cement-kiln-copilot-api:local --name cement-copilot
kind load docker-image cement-kiln-copilot-dashboard:local --name cement-copilot
kubectl apply -k deploy/k8s
```

### minikube

```bash
minikube start
eval "$(minikube docker-env)"
docker build -t cement-kiln-copilot-api:local .
docker build -f Dockerfile.streamlit -t cement-kiln-copilot-dashboard:local .
kubectl apply -k deploy/k8s
```

`kubectl apply -k` is the supported entry. The same manifests can be applied
without kustomize by passing each file except `kustomization.yaml`:

```bash
kubectl apply -f deploy/k8s/configmap.yaml \
  -f deploy/k8s/secret.example.yaml \
  -f deploy/k8s/api-deployment.yaml \
  -f deploy/k8s/api-service.yaml \
  -f deploy/k8s/dashboard-deployment.yaml \
  -f deploy/k8s/dashboard-service.yaml \
  -f deploy/k8s/mlflow-pvc.yaml \
  -f deploy/k8s/mlflow-deployment.yaml \
  -f deploy/k8s/mlflow-service.yaml
```

## Port-forward

```bash
kubectl port-forward svc/cement-copilot-api 8000:8000
kubectl port-forward svc/cement-copilot-dashboard 8501:8501
```

Optional tracking UI: `kubectl port-forward svc/cement-copilot-mlflow 5000:5000`.

The API reads `MODEL_PATH` and `DATABASE_URL` from the ConfigMap and Secret.
`/health` is the readiness and liveness probe. The dashboard's `API_BASE_URL`
is `http://cement-copilot-api:8000`. Mount a trained `joblib` file at
`/models/kiln_zone1_temp_model.joblib` before expecting predictions; the sample
volume is an emptyDir.
