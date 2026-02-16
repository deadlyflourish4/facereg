# --- VPC Network ---
resource "google_compute_network" "vpc" {
  name                    = "facereg-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "facereg-subnet"
  ip_cidr_range = "10.10.0.0/20"
  region        = var.region
  network       = google_compute_network.vpc.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.20.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.30.0.0/20"
  }
}

# --- GKE Cluster ---
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # Use separately managed node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }
}

# --- Node Pool ---
resource "google_container_node_pool" "primary_nodes" {
  name     = "facereg-nodes"
  location = var.zone
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = 50
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      app = "facereg"
    }

    tags = ["facereg-node"]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# --- Artifact Registry (Docker images) ---
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "facereg"
  format        = "DOCKER"
  description   = "FaceReg Docker images"
}

# --- Firewall ---
resource "google_compute_firewall" "allow_http" {
  name    = "facereg-allow-http"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["facereg-node"]
}
