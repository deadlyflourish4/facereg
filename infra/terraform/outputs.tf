output "cluster_name" {
  value = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  value     = google_container_cluster.primary.endpoint
  sensitive = true
}

output "region" {
  value = var.region
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "kubectl_config" {
  value = "gcloud container clusters get-credentials ${var.cluster_name} --zone ${var.zone} --project ${var.project_id}"
}
