# FaceReg — Face Recognition System

Real-time face recognition pipeline built with **FastAPI**, **YOLOv11**, **InsightFace (ArcFace)**, **Kafka**, and **Docker**.

## Architecture

```
Input (Image / Video / WebSocket)
       │
       ▼
[Face Detection]  →  YOLOv11n-face
       │
       ▼
[Quality Check]   →  Blur, brightness, pose (OpenCV)
       │
       ▼
[Anti-Spoofing]   →  LBP texture + FFT frequency + color analysis
       │                (Image-based: single frame)
       │                (Stream-based: multi-frame movement + temporal)
       ▼
[Face Alignment]  →  Affine transform → 112×112
       │
       ▼
[Face Embedding]  →  ArcFace (InsightFace) → 512D vector
       │
       ▼
[Recognition]     →  Cosine similarity (register / verify 1:1 / identify 1:N)
       │
       ▼
[Kafka Events]    →  face-events, face-alerts topics
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **API** | FastAPI + Uvicorn |
| **Detection** | YOLOv11 (Ultralytics) |
| **Recognition** | ArcFace via InsightFace |
| **Anti-Spoofing** | OpenCV texture/frequency analysis |
| **Storage** | JSON file (scalable to Vector DB) |
| **Events** | Apache Kafka (KRaft mode) |
| **Monitoring** | Kafka UI |
| **Deployment** | Docker + Docker Compose |
| **Cloud** | GKE (Google Kubernetes Engine) |
| **IaC** | Terraform (VPC, GKE, Artifact Registry) |
| **K8s Packaging** | Helm chart (Deployment, HPA, Ingress) |
| **CI/CD** | GitHub Actions (test → build → deploy) |

## API Endpoints

### REST (Image-based)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + module info |
| `POST` | `/detect` | Detect faces → bboxes, landmarks |
| `POST` | `/quality-check` | Quality assessment |
| `POST` | `/anti-spoof` | Single-image liveness check |
| `POST` | `/anti-spoof/video` | Multi-frame liveness (video upload) |
| `POST` | `/register` | Register user (multi-image) |
| `POST` | `/verify` | Verify 1:1 |
| `POST` | `/identify` | Identify 1:N |
| `GET` | `/users` | List registered users |
| `DELETE` | `/users/{id}` | Delete user |

### WebSocket (Stream-based)

| Endpoint | Description |
|----------|-------------|
| `WS /ws/detect` | Real-time face detection |
| `WS /ws/liveness` | Stream-based liveness check |
| `WS /ws/monitor` | Continuous identify (security camera) |

### SSE

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/stream/video` | Video file → frame-by-frame detection |

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (with Kafka)

```bash
docker compose up --build
```

Services:
- **FaceReg API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Kafka UI**: http://localhost:8080

## Kafka Events

All face events are published to Kafka topics:

| Topic | Events |
|-------|--------|
| `face-events` | detection, register, verify, identify |
| `face-alerts` | spoof_alert (potential spoofing detected) |

Event format:
```json
{
  "event_type": "detection",
  "timestamp": "2026-02-15T19:00:00",
  "data": {
    "total_faces": 2,
    "processing_time_ms": 45.2
  }
}
```

## GCloud Deployment (GKE)

### 1. Infrastructure (Terraform)

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID
terraform init
terraform plan
terraform apply
```

Creates: VPC → Subnet → GKE Cluster (auto-scaling) → Artifact Registry → Firewall

### 2. Deploy to K8s (Helm)

```bash
# Get cluster credentials
gcloud container clusters get-credentials facereg-cluster --zone asia-southeast1-a

# Deploy
helm upgrade --install facereg infra/helm/facereg \
  --set image.repository=asia-southeast1-docker.pkg.dev/YOUR_PROJECT/facereg/facereg \
  --set image.tag=latest
```

### 3. CI/CD (GitHub Actions)

Push to `main` triggers: **Lint → Test → Docker Build → Push to Artifact Registry → Helm Deploy to GKE**

Required GitHub Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`

## Project Structure

```
facereg/
├── app/
│   ├── main.py               # FastAPI app + routes
│   ├── config.py              # Environment config
│   ├── models/
│   │   ├── detector.py        # YOLOv11 face detection
│   │   ├── stream_detector.py # Real-time detection + frame skip
│   │   ├── quality.py         # Quality assessment (OpenCV)
│   │   ├── anti_spoof.py      # Anti-spoofing (texture + stream)
│   │   ├── aligner.py         # Face alignment (affine transform)
│   │   └── embedder.py        # ArcFace embedding (InsightFace)
│   ├── services/
│   │   ├── storage.py         # JSON file storage
│   │   ├── recognition.py     # Register / Verify / Identify
│   │   └── kafka_producer.py  # Kafka event publisher
│   ├── routes/
│   │   ├── detection.py       # /detect, /quality, /anti-spoof
│   │   ├── recognition.py     # /register, /verify, /identify
│   │   ├── users.py           # /users CRUD
│   │   └── websocket.py       # WebSocket + SSE endpoints
│   └── schemas/
│       └── responses.py       # Pydantic models
├── infra/
│   ├── terraform/             # GKE + VPC + Artifact Registry
│   │   ├── main.tf
│   │   ├── gke.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── helm/facereg/          # Kubernetes Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── ingress.yaml
│           ├── hpa.yaml
│           └── pvc.yaml
├── .github/workflows/
│   └── ci-cd.yaml             # GitHub Actions pipeline
├── weights/                   # Model weights
├── data/                      # Face database
├── Dockerfile
├── docker-compose.yaml        # App + Kafka + Kafka UI
├── requirements.txt
└── README.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.5 | YOLO detection confidence |
| `SPOOF_THRESHOLD` | 0.5 | Anti-spoofing threshold |
| `MATCH_THRESHOLD` | 0.5 | Verify similarity threshold |
| `IDENTIFY_THRESHOLD` | 0.4 | Identify minimum similarity |
| `DETECT_EVERY_N_FRAMES` | 3 | Frame skip for stream |
| `KAFKA_ENABLED` | false | Enable Kafka events |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Kafka address |